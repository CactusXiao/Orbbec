package main

import (
	"bufio"
	"context"
	"crypto/x509"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"
)

const origin = "https://orbbec-label.tail22716c.ts.net"
const tailnet = "tail22716c.ts.net"
const serverHost = "orbbec-label.tail22716c.ts.net"
const serverIP = "100.114.91.115"

var keyPattern = regexp.MustCompile(`^tskey-auth-[A-Za-z0-9_-]{20,512}$`)

type status struct {
	BackendState   string
	CurrentTailnet *struct{ MagicDNSSuffix string }
	Self           *struct{ Tags []string }
}

func isVendor(s status) bool {
	if s.CurrentTailnet == nil || s.CurrentTailnet.MagicDNSSuffix != tailnet || s.Self == nil {
		return false
	}
	for _, tag := range s.Self.Tags {
		if tag == "tag:vendor" {
			return true
		}
	}
	return false
}

func cliPath() (string, error) {
	var paths []string
	switch runtime.GOOS {
	case "darwin":
		paths = []string{"/Applications/Tailscale.app/Contents/MacOS/Tailscale"}
	case "windows":
		paths = []string{filepath.Join(os.Getenv("ProgramFiles"), "Tailscale", "tailscale.exe")}
	default:
		if p, e := exec.LookPath("tailscale"); e == nil {
			return p, nil
		}
	}
	for _, p := range paths {
		if s, e := os.Stat(p); e == nil && !s.IsDir() {
			return p, nil
		}
	}
	return "", errors.New("请先安装并打开 Tailscale：https://tailscale.com/download；无需登录 Tailscale 账号")
}

func runCLI(cli string, args ...string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 75*time.Second)
	defer cancel()
	return exec.CommandContext(ctx, cli, args...).CombinedOutput()
}

func readStatus(cli string) (status, error) {
	var s status
	b, e := runCLI(cli, "status", "--json")
	// Some clients return a nonzero code while logged out but still emit JSON.
	if err := json.Unmarshal(b, &s); err != nil {
		return s, fmt.Errorf("无法读取 Tailscale 状态，请先打开客户端并完成系统网络授权：%v", e)
	}
	return s, nil
}

// The operator may explicitly bundle a key. Never put it in process arguments or logs.
func bundledKey(executable string) (string, error) {
	p := filepath.Join(filepath.Dir(executable), "..", "network-key.txt")
	f, e := os.Open(p)
	if os.IsNotExist(e) {
		return "", nil
	}
	if e != nil {
		return "", errors.New("无法读取包内网络令牌")
	}
	defer f.Close()
	b, e := io.ReadAll(io.LimitReader(f, 1024))
	if e != nil {
		return "", errors.New("无法读取包内网络令牌")
	}
	key := strings.TrimSpace(string(b))
	if !keyPattern.MatchString(key) {
		return "", errors.New("包内网络令牌格式错误，请联系管理员更新交付包")
	}
	return key, nil
}

func keyFile(key string) (string, func(), error) {
	dir, e := os.MkdirTemp("", "orbbec-enroll-")
	if e != nil {
		return "", nil, e
	}
	var once sync.Once
	clean := func() { once.Do(func() { os.RemoveAll(dir) }) }
	if e = secureDir(dir); e != nil {
		clean()
		return "", nil, e
	}
	p := filepath.Join(dir, "auth-key")
	if e = os.WriteFile(p, []byte(key), 0600); e != nil {
		clean()
		return "", nil, e
	}
	return p, clean, nil
}

func enrollmentArgs(path string) []string {
	return []string{"up", "--auth-key=file:" + path, "--advertise-tags=tag:vendor", "--accept-dns=true", "--accept-routes=false", "--shields-up", "--timeout=60s"}
}

func entranceClient(direct bool) *http.Client {
	dialer := &net.Dialer{Timeout: 8 * time.Second}
	transport := &http.Transport{Proxy: nil, TLSHandshakeTimeout: 12 * time.Second}
	transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		// Diagnostic only: bypass DNS while keeping the HTTPS hostname and certificate checks.
		if direct && addr == serverHost+":443" {
			addr = serverIP + ":443"
		}
		return dialer.DialContext(ctx, network, addr)
	}
	return &http.Client{Timeout: 25 * time.Second, Transport: transport, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
}

func entranceRequest(client *http.Client, url string) error {
	response, e := client.Get(url)
	if e != nil {
		return e
	}
	defer response.Body.Close()
	io.Copy(io.Discard, io.LimitReader(response.Body, 4096))
	if response.StatusCode != http.StatusUnauthorized {
		return fmt.Errorf("HTTP %d（未登录检查应返回 401）", response.StatusCode)
	}
	return nil
}

func connectionError(e error) string {
	var dns *net.DNSError
	var unknown x509.UnknownAuthorityError
	var invalid x509.CertificateInvalidError
	var hostname x509.HostnameError
	var network net.Error
	switch {
	case errors.As(e, &dns):
		return "DNS 无法解析工作台域名；检查 Tailscale DNS 设置。"
	case errors.As(e, &unknown), errors.As(e, &invalid), errors.As(e, &hostname):
		return "HTTPS 证书验证失败；检查系统时间和证书信任，不要跳过验证。"
	case errors.As(e, &network) && network.Timeout():
		return "连接超时；检查 Tailscale 网络、防火墙或代理冲突。"
	default:
		return "连接失败：" + e.Error()
	}
}

func checkEntrance() error {
	client := entranceClient(false)
	defer client.CloseIdleConnections()
	var e error
	for attempt := 1; attempt <= 3; attempt++ {
		fmt.Printf("检查工作台连接（%d/3）……\n", attempt)
		e = entranceRequest(client, origin+"/api/identity")
		if e == nil {
			return nil
		}
		if attempt < 3 {
			time.Sleep(time.Duration(attempt*2) * time.Second)
		}
	}
	return errors.New("设备已接入。" + connectionError(e) + " 请运行“连接诊断”，无需重新输入令牌。")
}

func diagnose() error {
	lines := []string{"Orbbec 连接诊断", time.Now().Format(time.RFC3339), "系统：" + runtime.GOOS + "/" + runtime.GOARCH}
	add := func(s string) { lines = append(lines, s); fmt.Println(s) }
	cli, e := cliPath()
	if e != nil {
		add(e.Error())
	} else {
		s, err := readStatus(cli)
		if err != nil {
			add(err.Error())
		} else {
			add(fmt.Sprintf("Tailscale 状态：%s；外包标签及网络匹配：%t", s.BackendState, isVendor(s)))
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	ips, e := net.DefaultResolver.LookupHost(ctx, serverHost)
	cancel()
	if e != nil {
		add("域名解析：" + connectionError(e))
	} else {
		add("域名解析：" + strings.Join(ips, ", "))
	}
	conn, e := net.DialTimeout("tcp", serverIP+":443", 8*time.Second)
	if e != nil {
		add("虚拟内网 TCP 443：" + e.Error())
	} else {
		conn.Close()
		add("虚拟内网 TCP 443：成功")
	}
	for _, direct := range []bool{false, true} {
		name := "域名 HTTPS"
		if direct {
			name = "固定内网 IP 的 HTTPS（仍校验证书）"
		}
		client := entranceClient(direct)
		e = entranceRequest(client, origin+"/api/identity")
		client.CloseIdleConnections()
		if e != nil {
			add(name + "：" + connectionError(e) + " [" + e.Error() + "]")
		} else {
			add(name + "：成功，401 未登录响应正常")
		}
	}
	exe, e := os.Executable()
	if e != nil {
		return e
	}
	p := filepath.Join(filepath.Dir(exe), "..", "connection-diagnostics.txt")
	if e = os.WriteFile(p, []byte(strings.Join(lines, "\n")+"\n"), 0600); e != nil {
		return fmt.Errorf("诊断文件保存失败：%v", e)
	}
	fmt.Println("诊断已保存：", p)
	return nil
}

func connect(checkOnly bool) error {
	cli, e := cliPath()
	if e != nil {
		return e
	}
	s, e := readStatus(cli)
	if e != nil {
		return e
	}
	if s.BackendState == "NeedsMachineAuth" {
		return errors.New("设备等待管理员审批，请联系管理员；不要重复输入令牌")
	}
	if checkOnly {
		if !isVendor(s) || s.BackendState != "Running" {
			return errors.New("尚未通过外包设备令牌接入")
		}
		return checkEntrance()
	}
	if s.BackendState == "Running" && isVendor(s) {
		return checkEntrance()
	}
	if s.CurrentTailnet != nil && !isVendor(s) {
		return errors.New("这台电脑已绑定其他 Tailscale 身份。请先确认可切换，再在客户端退出该身份；工具不会替换现有连接")
	}
	if isVendor(s) && s.BackendState == "Stopped" {
		if _, e = runCLI(cli, "up", "--timeout=60s"); e != nil {
			return errors.New("无法恢复连接，请打开 Tailscale 检查状态")
		}
	} else {
		if s.BackendState != "NeedsLogin" && s.BackendState != "Stopped" {
			return fmt.Errorf("请先完成 Tailscale 系统授权（%s）", s.BackendState)
		}
		executable, e := os.Executable()
		if e != nil {
			return errors.New("无法定位接入工具")
		}
		key, e := bundledKey(executable)
		if e != nil {
			return e
		}
		restore := func() {}
		if key == "" {
			fmt.Println("粘贴管理员分配的网络接入令牌，按回车（输入不显示）：")
			restore, e = hideInput()
			if e != nil {
				return errors.New("无法启用隐藏输入，请在终端窗口运行接入工具")
			}
		} else {
			fmt.Println("已读取包内网络令牌。")
		}
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt)
		done := make(chan struct{})
		var cleanupMu sync.Mutex
		cleanup := func() {}
		go func() {
			select {
			case <-sig:
				restore()
				cleanupMu.Lock()
				cleanup()
				cleanupMu.Unlock()
				os.Exit(130)
			case <-done:
			}
		}()
		defer func() { signal.Stop(sig); close(done); restore() }()
		if key == "" {
			reader := bufio.NewReader(io.LimitReader(os.Stdin, 1024))
			var readErr error
			key, readErr = reader.ReadString('\n')
			restore()
			fmt.Println()
			if readErr != nil {
				return errors.New("未读取到完整令牌")
			}
		}
		key = strings.TrimSpace(key)
		if !keyPattern.MatchString(key) {
			return errors.New("令牌格式不正确，请粘贴完整的 tskey-auth- 开头令牌")
		}
		path, remove, e := keyFile(key)
		if e != nil {
			return errors.New("无法创建私有临时文件，接入已取消")
		}
		cleanupMu.Lock()
		cleanup = remove
		cleanupMu.Unlock()
		defer remove()
		fmt.Println("正在接入……")
		_, e = runCLI(cli, enrollmentArgs(path)...)
		// Raw CLI output can contain authentication details; never display it.
		remove()
		key = ""
		if e != nil {
			return errors.New("接入未完成。请检查令牌是否过期／已撤销及系统授权；已接入设备无需重新输入令牌")
		}
	}
	s, e = readStatus(cli)
	if e != nil {
		return e
	}
	if !isVendor(s) || s.BackendState != "Running" {
		return errors.New("设备未取得项目外包权限，请管理员检查令牌标签与设备审批")
	}
	return checkEntrance()
}

func main() {
	checkOnly := flag.Bool("check", false, "仅检查接入状态")
	noOpen := flag.Bool("no-open", false, "不打开浏览器")
	diagnostic := flag.Bool("diagnose", false, "仅诊断连接，不更改账号或网络配置")
	flag.Parse()
	fmt.Println("Orbbec 网络接入｜Tailscale 令牌")
	if *diagnostic {
		if e := diagnose(); e != nil {
			fmt.Println(e)
		}
		if !*noOpen {
			fmt.Println("按回车关闭。")
			bufio.NewReader(os.Stdin).ReadString('\n')
		}
		return
	}
	e := connect(*checkOnly)
	if e != nil {
		fmt.Println(e)
	} else {
		fmt.Println("网络已连接。QC／Label 请使用自己的工作台账号登录。")
	}
	if e == nil && !*noOpen && !*checkOnly {
		switch runtime.GOOS {
		case "darwin":
			exec.Command("open", origin).Run()
		case "windows":
			exec.Command("rundll32", "url.dll,FileProtocolHandler", origin).Run()
		default:
			exec.Command("xdg-open", origin).Run()
		}
	}
	if !*checkOnly {
		fmt.Println("按回车关闭。")
		bufio.NewReader(os.Stdin).ReadString('\n')
	}
	if e != nil {
		os.Exit(1)
	}
}
