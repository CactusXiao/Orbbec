package main

import (
	"fmt"
	"os"
	"os/exec"
	"os/user"
	"sync"
	"syscall"
	"unsafe"
)

func init() {
	dll := syscall.NewLazyDLL("kernel32.dll")
	dll.NewProc("SetConsoleOutputCP").Call(65001)
	dll.NewProc("SetConsoleCP").Call(65001)
}

func secureDir(path string) error {
	u, e := user.Current()
	if e != nil {
		return e
	}
	// Remove inherited access before creating the secret file; grant only this SID.
	return exec.Command("icacls", path, "/inheritance:r", "/grant:r", "*"+u.Uid+":(OI)(CI)F").Run()
}
func hideInput() (func(), error) {
	dll := syscall.NewLazyDLL("kernel32.dll")
	get := dll.NewProc("GetConsoleMode")
	set := dll.NewProc("SetConsoleMode")
	var mode uint32
	h := os.Stdin.Fd()
	if ok, _, e := get.Call(h, uintptr(unsafe.Pointer(&mode))); ok == 0 {
		return nil, fmt.Errorf("console: %v", e)
	}
	if ok, _, e := set.Call(h, uintptr(mode&^4)); ok == 0 {
		return nil, fmt.Errorf("console: %v", e)
	}
	var once sync.Once
	return func() { once.Do(func() { set.Call(h, uintptr(mode)) }) }, nil
}
