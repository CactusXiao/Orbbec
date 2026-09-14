// A fixed-destination HTTPS gateway and vendor launcher. No SSH or external modules.
package main

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"strings"
	"time"
)

//go:embed server.pem
var certificate []byte

const remoteOrigin = "https://10.162.241.5:18884"
const localAddress = "127.0.0.1:18885"
const localOrigin = "http://" + localAddress
const backendOrigin = "http://127.0.0.1:18882"
const backendCookie = "orbbec_account_18882"
const launcherCookie = "orbbec_account_18885"
const healthPath = "/_orbbec_launcher_health"

func fingerprint() string { sum := sha256.Sum256(certificate); return hex.EncodeToString(sum[:]) }
func transport(secure bool) *http.Transport {
	t := http.DefaultTransport.(*http.Transport).Clone()
	// Do not send private VPN traffic or credentials to environment-configured proxies.
	t.Proxy = nil
	t.DialContext = (&net.Dialer{Timeout: 8 * time.Second, KeepAlive: 30 * time.Second}).DialContext
	t.ResponseHeaderTimeout = 90 * time.Second
	t.MaxIdleConnsPerHost = 16
	t.MaxConnsPerHost = 64
	if secure {
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(certificate) {
			panic("invalid embedded server certificate")
		}
		t.TLSClientConfig = &tls.Config{RootCAs: pool, MinVersion: tls.VersionTLS12}
	}
	return t
}
func proxyHandler(expectedOrigin, targetOrigin string, tr http.RoundTripper, secureCookie, local bool) http.Handler {
	target, err := url.Parse(targetOrigin)
	if err != nil {
		panic(err)
	}
	expected, err := url.Parse(expectedOrigin)
	if err != nil {
		panic(err)
	}
	proxy := &httputil.ReverseProxy{
		Rewrite: func(p *httputil.ProxyRequest) {
			p.SetURL(target)
			p.Out.Host = target.Host
			p.Out.Header.Del("Forwarded")
			if p.In.Header.Get("Origin") != "" {
				p.Out.Header.Set("Origin", targetOrigin)
			}
			p.Out.Header.Del("Referer")
			cookieName := backendCookie
			if local {
				cookieName = launcherCookie
			}
			p.Out.Header.Del("Cookie")
			if cookie, err := p.In.Cookie(cookieName); err == nil {
				p.Out.AddCookie(&http.Cookie{Name: backendCookie, Value: cookie.Value})
			}
		},
		Transport: tr, FlushInterval: 100 * time.Millisecond,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, e error) {
			w.Header().Set("Cache-Control", "no-store")
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, "<!doctype html><meta charset=utf-8><title>连接暂时中断</title><h2>暂时无法连接工作台</h2><p>请确认企业 VPN 已连接，再刷新页面。浏览器草稿仍保留。</p>")
		},
		ModifyResponse: func(r *http.Response) error {
			cookies := r.Cookies()
			r.Header.Del("Set-Cookie")
			for _, c := range cookies {
				if local && c.Name == backendCookie {
					c.Name = launcherCookie
				}
				c.Secure = secureCookie
				r.Header.Add("Set-Cookie", c.String())
			}
			// If the upstream introduces a same-origin redirect, keep it within this entry.
			if location := r.Header.Get("Location"); location != "" {
				u, e := url.Parse(location)
				if e == nil && u.Host == target.Host && u.Scheme == target.Scheme {
					u.Host = expected.Host
					u.Scheme = expected.Scheme
					r.Header.Set("Location", u.String())
				}
			}
			return nil
		},
	}
	slots := make(chan struct{}, 64)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		if r.Host != expected.Host || r.URL.IsAbs() || strings.HasPrefix(r.RequestURI, "//") {
			http.Error(w, "Invalid entry address", 403)
			return
		}
		if origin := r.Header.Get("Origin"); origin != "" && origin != expectedOrigin {
			http.Error(w, "Cross-origin request rejected", 403)
			return
		}
		if r.Header.Get("Sec-Fetch-Site") == "cross-site" && r.Header.Get("Sec-Fetch-Mode") != "navigate" {
			http.Error(w, "Cross-site request rejected", 403)
			return
		}
		if r.Method != "GET" && r.Method != "POST" && r.Method != "HEAD" {
			http.Error(w, "Method not allowed", 405)
			return
		}
		if local && r.URL.Path == healthPath && r.Method == "GET" {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Cache-Control", "no-store")
			json.NewEncoder(w).Encode(map[string]string{"app": "orbbec-vpn-launcher-v1", "certificate": fingerprint()})
			return
		}
		if r.ContentLength > 32*1024*1024 {
			http.Error(w, "Request too large", 413)
			return
		}
		// The backend requires an explicit Content-Length for JSON submissions.
		if r.Method == "POST" && r.ContentLength <= 0 {
			http.Error(w, "Content-Length required", 411)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, 32*1024*1024)
		select {
		case slots <- struct{}{}:
			defer func() { <-slots }()
		default:
			http.Error(w, "Busy, please retry", 503)
			return
		}
		proxy.ServeHTTP(w, r)
	})
}
func openBrowser() {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", localOrigin+"/")
	case "darwin":
		cmd = exec.Command("open", localOrigin+"/")
	default:
		cmd = exec.Command("xdg-open", localOrigin+"/")
	}
	if err := cmd.Start(); err != nil {
		fmt.Println("请手动打开：", localOrigin+"/")
	}
}
func runningLauncher() bool {
	c := &http.Client{Transport: transport(false), Timeout: 2 * time.Second}
	resp, err := c.Get(localOrigin + healthPath)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	var status map[string]string
	return resp.StatusCode == 200 && json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&status) == nil && status["app"] == "orbbec-vpn-launcher-v1" && status["certificate"] == fingerprint()
}
func validateTailnetOrigin(value string) error {
	u, err := url.Parse(value)
	if err != nil || u.Scheme != "https" || u.User != nil || u.Path != "" || u.RawQuery != "" || u.Fragment != "" || !strings.HasSuffix(u.Hostname(), ".ts.net") || u.Port() != "" {
		return errors.New("tailnet origin must be an HTTPS *.ts.net origin without port, credentials or path")
	}
	return nil
}

func main() {
	serverMode := flag.Bool("server", false, "Run the laboratory HTTPS gateway")
	tailnetClient := flag.Bool("tailscale", false, "Connect the local launcher through Tailscale HTTPS")
	tailnetOrigin := flag.String("tailnet-origin", "", "Serve a fixed Tailscale HTTPS origin through a loopback adapter")
	certFile := flag.String("cert", "server.pem", "Server certificate path")
	keyFile := flag.String("key", "server.key", "Server private key path")
	noOpen := flag.Bool("no-open", false, "Do not open the browser automatically")
	check := flag.Bool("check", false, "Check the VPN backend without signing in")
	flag.Parse()
	if *tailnetClient && (*serverMode || *tailnetOrigin != "" || *check) {
		log.Fatal("tailscale cannot be combined with server/tailnet-origin/check")
	}
	if *tailnetOrigin != "" {
		if err := validateTailnetOrigin(*tailnetOrigin); err != nil {
			log.Fatal(err)
		}
		if *serverMode || *check {
			log.Fatal("tailnet-origin cannot be combined with server/check")
		}
	}
	if *check {
		c := &http.Client{Transport: transport(true), Timeout: 12 * time.Second, CheckRedirect: func(_ *http.Request, _ []*http.Request) error { return errors.New("unexpected redirect") }}
		r, e := c.Get(remoteOrigin + "/api/identity")
		if e != nil {
			fmt.Fprintln(os.Stderr, "VPN 后端连接失败：", e)
			os.Exit(1)
		}
		defer r.Body.Close()
		if r.StatusCode != 401 {
			fmt.Fprintln(os.Stderr, "工作台返回异常状态：", r.StatusCode)
			os.Exit(1)
		}
		fmt.Println("VPN 直连成功；HTTPS 证书校验通过；工作台要求账号登录。")
		return
	}
	address := localAddress
	handler := proxyHandler(localOrigin, remoteOrigin, transport(true), false, true)
	if *tailnetClient {
		tr := transport(false)
		tr.DialContext = (&net.Dialer{Timeout: 30 * time.Second, KeepAlive: 30 * time.Second}).DialContext
		tr.TLSHandshakeTimeout = 30 * time.Second
		handler = proxyHandler(localOrigin, "https://orbbec-label.tail22716c.ts.net", tr, false, true)
	}
	if *serverMode {
		address = "10.162.241.5:18884"
		handler = proxyHandler(remoteOrigin, backendOrigin, transport(false), true, false)
	}
	if *tailnetOrigin != "" {
		address = "127.0.0.1:18886"
		handler = proxyHandler(*tailnetOrigin, backendOrigin, transport(false), true, false)
	}
	listener, err := net.Listen("tcp4", address)
	if err != nil {
		if !*serverMode && *tailnetOrigin == "" && runningLauncher() {
			if !*noOpen {
				openBrowser()
			}
			fmt.Println("工作台已在运行。")
			return
		}
		fmt.Fprintln(os.Stderr, "无法启动，请确认同一工作台未被其他程序占用：", err)
		os.Exit(1)
	}
	srv := &http.Server{Handler: handler, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 60 * time.Second, IdleTimeout: 90 * time.Second, MaxHeaderBytes: 32 * 1024, TLSConfig: &tls.Config{MinVersion: tls.VersionTLS12}}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		srv.Shutdown(shutdown)
	}()
	if *serverMode {
		log.Println("Orbbec VPN HTTPS gateway ready on", address)
		err = srv.ServeTLS(listener, *certFile, *keyFile)
	} else if *tailnetOrigin != "" {
		log.Println("Orbbec Tailscale loopback adapter ready on", address)
		err = srv.Serve(listener)
	} else {
		if *tailnetClient {
			fmt.Println("Orbbec 外包工作台\n请保持 Tailscale 已连接。\n浏览器入口：" + localOrigin + "/")
		} else {
			fmt.Println("Orbbec 外包工作台\n请连接企业 VPN，并保留此窗口。退出时按 Ctrl+C。\n浏览器入口：" + localOrigin + "/\n本程序直接通过 HTTPS 连接后端，不使用 SSH。")
		}
		if !*noOpen {
			openBrowser()
		}
		err = srv.Serve(listener)
	}
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
}
