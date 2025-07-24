package main

import (
    "fmt"
    "net"
    "os"
    "os/exec"
    "runtime"
)

func main() {
    // Hostname
    hostname, err := os.Hostname()
    if err != nil {
        fmt.Printf("Hostname: unknown (error: %v)\n", err)
    } else {
        fmt.Printf("Hostname: %s\n", hostname)
    }

    // IP addresses
    addrs, err := net.InterfaceAddrs()
    if err != nil {
        fmt.Printf("IP Address: unknown (error: %v)\n", err)
    } else {
        for _, addr := range addrs {
            if ipNet, ok := addr.(*net.IPNet); ok && !ipNet.IP.IsLoopback() {
                if ipNet.IP.To4() != nil {
                    fmt.Printf("IP Address: %s\n", ipNet.IP.String())
                }
            }
        }
    }

    // OS type
    fmt.Printf("OS Type: %s\n", runtime.GOOS)

    // OS version via uname -r
    unameOut, err := exec.Command("uname", "-r").Output()
    if err != nil {
        fmt.Printf("OS Version: unknown (error: %v)\n", err)
    } else {
        fmt.Printf("OS Version: %s\n", string(unameOut))
    }
}

