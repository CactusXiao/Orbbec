//go:build !windows

package main

import (
	"os"
	"os/exec"
	"strings"
	"sync"
)

func secureDir(path string) error { return os.Chmod(path, 0700) }
func hideInput() (func(), error) {
	get := exec.Command("stty", "-g")
	get.Stdin = os.Stdin
	state, e := get.Output()
	if e != nil {
		return nil, e
	}
	cmd := exec.Command("stty", "-echo")
	cmd.Stdin = os.Stdin
	if e = cmd.Run(); e != nil {
		return nil, e
	}
	var once sync.Once
	return func() {
		once.Do(func() { cmd := exec.Command("stty", strings.TrimSpace(string(state))); cmd.Stdin = os.Stdin; cmd.Run() })
	}, nil
}
