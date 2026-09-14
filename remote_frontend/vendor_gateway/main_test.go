package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGatewayLimitsAndOrigin(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Origin") != "http://"+r.Host {
			t.Error("origin not rewritten to fixed backend")
		}
		if r.Header.Get("Forwarded") != "" || r.Header.Get("X-Forwarded-Host") != "" {
			t.Error("untrusted proxy headers forwarded")
		}
		http.SetCookie(w, &http.Cookie{Name: backendCookie, Value: "token", HttpOnly: true, SameSite: http.SameSiteStrictMode, Path: "/"})
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"ok":true}`)
	}))
	defer upstream.Close()
	h := proxyHandler(remoteOrigin, upstream.URL, transport(false), true, false)
	for _, tt := range []struct {
		method, host, origin string
		size                 int64
		want                 int
	}{
		{"POST", "10.162.241.5:18884", remoteOrigin, 2, 200},
		{"POST", "10.162.241.5:18884", "https://evil.example", 2, 403},
		{"GET", "evil.example", "", 0, 403},
		{"CONNECT", "10.162.241.5:18884", "", 0, 405},
		{"POST", "10.162.241.5:18884", remoteOrigin, 33 * 1024 * 1024, 413},
		{"POST", "10.162.241.5:18884", remoteOrigin, -1, 411},
	} {
		r := httptest.NewRequest(tt.method, "/api/login", strings.NewReader("{}"))
		r.Host = tt.host
		r.ContentLength = tt.size
		r.Header.Set("Origin", tt.origin)
		r.Header.Set("Forwarded", "for=evil")
		r.Header.Set("X-Forwarded-Host", "evil")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != tt.want {
			t.Fatalf("%+v: got %d", tt, w.Code)
		}
		if w.Code == 200 {
			c := w.Result().Cookies()[0]
			if !c.Secure || !c.HttpOnly || c.SameSite != http.SameSiteStrictMode {
				t.Fatal("unsafe gateway cookie")
			}
		}
	}
}
func TestLauncherStreamingAndCookies(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "bytes=2-4" {
			t.Error("lost range header")
		}
		http.SetCookie(w, &http.Cookie{Name: backendCookie, Value: "token", Secure: true, HttpOnly: true, SameSite: http.SameSiteStrictMode, Path: "/"})
		w.Header().Set("Content-Range", "bytes 2-4/10")
		w.WriteHeader(206)
		io.WriteString(w, "234")
	}))
	defer upstream.Close()
	h := proxyHandler(localOrigin, upstream.URL, transport(false), false, true)
	r := httptest.NewRequest("GET", "/media/video.mp4", nil)
	r.Host = localAddress
	r.Header.Set("Range", "bytes=2-4")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 206 || w.Body.String() != "234" || w.Header().Get("Content-Range") != "bytes 2-4/10" {
		t.Fatal("range response changed")
	}
	c := w.Result().Cookies()[0]
	if c.Name != launcherCookie || c.Secure || !c.HttpOnly || c.SameSite != http.SameSiteStrictMode {
		t.Fatal("incorrect loopback cookie")
	}
	r = httptest.NewRequest("GET", healthPath, nil)
	r.Host = localAddress
	w = httptest.NewRecorder()
	h.ServeHTTP(w, r)
	var status map[string]string
	json.Unmarshal(w.Body.Bytes(), &status)
	if status["certificate"] != fingerprint() {
		t.Fatal("launcher identity missing")
	}
}
func TestTLSRejectsOtherServer(t *testing.T) {
	s := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "wrong server") }))
	defer s.Close()
	c := &http.Client{Transport: transport(true)}
	if r, err := c.Get(s.URL); err == nil {
		r.Body.Close()
		t.Fatal("untrusted TLS server accepted")
	}
}

func TestLauncherCookieIsolation(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := r.Cookie(backendCookie)
		if err != nil || c.Value != "new-entry" || len(r.Cookies()) != 1 {
			t.Error("cookies from other localhost ports were forwarded")
		}
		w.WriteHeader(204)
	}))
	defer upstream.Close()
	h := proxyHandler(localOrigin, upstream.URL, transport(false), false, true)
	r := httptest.NewRequest("GET", "/api/identity", nil)
	r.Host = localAddress
	r.AddCookie(&http.Cookie{Name: backendCookie, Value: "old-entry"})
	r.AddCookie(&http.Cookie{Name: launcherCookie, Value: "new-entry"})
	w := httptest.NewRecorder()
	h.ServeHTTP(w, r)
	if w.Code != 204 {
		t.Fatal(w.Code)
	}
}

func TestTailnetOriginValidation(t *testing.T) {
	for _, v := range []string{"http://lab.tail.ts.net", "https://lab.example.com", "https://u:p@lab.tail.ts.net", "https://lab.tail.ts.net/", "https://lab.tail.ts.net?x=1", "https://lab.tail.ts.net:443"} {
		if validateTailnetOrigin(v) == nil {
			t.Fatalf("unsafe origin accepted: %s", v)
		}
	}
	if err := validateTailnetOrigin("https://orbbec-label.tail-example.ts.net"); err != nil {
		t.Fatal(err)
	}
}
