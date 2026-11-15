#!/usr/bin/env bash
# Hardened, key-only SSH setup for this machine.
# Usage:  sudo bash setup-ssh-hardened.sh [CLIENT_IP]
#   CLIENT_IP (optional): only this IP may reach port 22. Without it, port 22 is rate-limited.
set -euo pipefail

[[ $EUID -eq 0 ]] || { echo "Run with sudo."; exit 1; }

TARGET_USER="intel"
AUTH_KEYS="/home/$TARGET_USER/.ssh/authorized_keys"
CLIENT_IP="${1:-}"

# Refuse to continue without a key, or you'd lock yourself out.
if ! grep -qE '^(ssh-ed25519|ssh-rsa|ecdsa-sha2-|sk-ssh-ed25519|sk-ecdsa-sha2-)' "$AUTH_KEYS" 2>/dev/null; then
  echo "No public key in $AUTH_KEYS. Paste your client's public key there first."
  exit 1
fi
chown -R "$TARGET_USER:$TARGET_USER" "/home/$TARGET_USER/.ssh"
chmod 700 "/home/$TARGET_USER/.ssh"; chmod 600 "$AUTH_KEYS"

echo "== Firewall"
ufw default deny incoming
ufw default allow outgoing
if [[ -n "$CLIENT_IP" ]]; then
  ufw allow from "$CLIENT_IP" to any port 22 proto tcp
else
  ufw limit 22/tcp
fi
ufw --force enable

echo "== sshd hardening"
cat > /etc/ssh/sshd_config.d/00-hardening.conf <<EOF
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
AuthenticationMethods publickey
PermitEmptyPasswords no
AllowUsers $TARGET_USER
MaxAuthTries 3
LoginGraceTime 20
X11Forwarding no
AllowAgentForwarding no
AllowTcpForwarding no
ClientAliveInterval 300
ClientAliveCountMax 2
EOF
sshd -t

echo "== fail2ban"
cat > /etc/fail2ban/jail.d/sshd.local <<'EOF'
[sshd]
enabled = true
backend = systemd
maxretry = 3
bantime = 1h
findtime = 10m
EOF
systemctl enable --now fail2ban
systemctl restart fail2ban

echo "== Start SSH"
systemctl enable --now ssh
systemctl restart ssh

echo "== Effective settings"
sshd -T | grep -Ei '^(passwordauthentication|permitrootlogin|kbdinteractiveauthentication|authenticationmethods|allowusers) '
ufw status | grep -E '22|Status'
echo "Done. Test from your client: ssh $TARGET_USER@$(hostname -I | awk '{print $1}')"
