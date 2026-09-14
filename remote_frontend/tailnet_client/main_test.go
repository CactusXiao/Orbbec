package main

import (
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEntranceResponseAndRedirect(t *testing.T) {
	for _, code := range []int{401, 200, 302, 502} {
		t.Run(http.StatusText(code), func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if code == 302 {
					w.Header().Set("Location", "https://example.invalid/")
				}
				w.WriteHeader(code)
			}))
			defer server.Close()
			client := entranceClient(false)
			defer client.CloseIdleConnections()
			err := entranceRequest(client, server.URL)
			if (err == nil) != (code == 401) {
				t.Fatalf("status %d: %v", code, err)
			}
		})
	}
}

func TestDiagnosticKeepsCertificateVerification(t *testing.T) {
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(401) }))
	defer server.Close()
	for _, direct := range []bool{false, true} {
		client := entranceClient(direct)
		defer client.CloseIdleConnections()
		err := entranceRequest(client, server.URL)
		if err == nil || !strings.Contains(connectionError(err), "证书验证失败") {
			t.Fatalf("certificate check bypassed: %v", err)
		}
	}
	err := &url.Error{Op: "Get", URL: "https://example.invalid", Err: &net.DNSError{Err: "no such host", Name: "example.invalid"}}
	if !strings.Contains(connectionError(err), "DNS") {
		t.Fatal("DNS failure not distinguished")
	}
}

func TestKeyFileAndArguments(t *testing.T) {
	key := "tskey-auth-TESTONLY00000000000000000000"
	p, clean, e := keyFile(key)
	if e != nil {
		t.Fatal(e)
	}
	defer clean()
	b, e := os.ReadFile(p)
	if e != nil || string(b) != key {
		t.Fatal("key file")
	}
	args := strings.Join(enrollmentArgs(p), " ")
	if strings.Contains(args, key) || !strings.Contains(args, "--auth-key=file:") || !strings.Contains(args, "--advertise-tags=tag:vendor") {
		t.Fatal("key exposed in arguments")
	}
	for _, bad := range []string{"--reset", "--force-reauth", "--advertise-exit-node"} {
		if strings.Contains(args, bad) {
			t.Fatal("unsafe enrollment flag")
		}
	}
	clean()
	if _, e = os.Stat(filepath.Dir(p)); !os.IsNotExist(e) {
		t.Fatal("secret not removed")
	}
}

func TestRejectWrongIdentity(t *testing.T) {
	s := status{}
	if isVendor(s) {
		t.Fatal("empty identity accepted")
	}
	s.CurrentTailnet = &struct{ MagicDNSSuffix string }{tailnet}
	s.Self = &struct{ Tags []string }{[]string{"tag:vendor"}}
	if !isVendor(s) {
		t.Fatal("vendor rejected")
	}
	s.CurrentTailnet.MagicDNSSuffix = "other.ts.net"
	if isVendor(s) {
		t.Fatal("foreign network accepted")
	}
	s.CurrentTailnet.MagicDNSSuffix = tailnet
	s.Self.Tags = nil
	if isVendor(s) {
		t.Fatal("personal identity accepted")
	}
}

func TestBundledKey(t *testing.T) {
	dir := t.TempDir()
	os.Mkdir(filepath.Join(dir, "bin"), 0700)
	bin := filepath.Join(dir, "bin", "network")
	key, e := bundledKey(bin)
	if e != nil || key != "" {
		t.Fatal("missing key should allow prompt")
	}
	p := filepath.Join(dir, "network-key.txt")
	want := "tskey-auth-TESTONLY00000000000000000000"
	if e = os.WriteFile(p, []byte(want+"\r\n"), 0600); e != nil {
		t.Fatal(e)
	}
	key, e = bundledKey(bin)
	if e != nil || key != want {
		t.Fatal("bundled key not loaded")
	}
	os.WriteFile(p, []byte("invalid"), 0600)
	if _, e = bundledKey(bin); e == nil {
		t.Fatal("invalid bundle accepted")
	}
}
