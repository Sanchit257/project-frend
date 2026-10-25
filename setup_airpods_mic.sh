#!/bin/bash
# Makes Bluetooth headset microphones (AirPods) work on the Raspberry Pi.
#
# The Pi's Broadcom Bluetooth chip sends call ("SCO") audio to hardware PCM
# pins by default, so Linux receives pure silence from headset mics. This
# vendor command (Write_SCO_PCM_Int_Param, routing = HCI) sends that audio to
# Linux instead. The chip forgets it on every boot, so it is also installed
# as a small systemd service.
#
# Run once:   sudo ./setup_airpods_mic.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

cat > /etc/systemd/system/bt-sco-over-hci.service <<'EOF'
[Unit]
Description=Route Bluetooth SCO (headset mic) audio over HCI
After=bluetooth.service hciuart.service
Requires=bluetooth.service

[Service]
Type=oneshot
ExecStartPre=/bin/sleep 2
ExecStart=/usr/bin/hcitool cmd 0x3f 0x1c 0x01 0x02 0x00 0x01 0x01
RemainAfterExit=yes

[Install]
WantedBy=bluetooth.target
EOF

systemctl daemon-reload
systemctl enable --now bt-sco-over-hci.service
systemctl restart bt-sco-over-hci.service
echo "Done. SCO audio now goes over HCI (and will after every reboot)."
echo "Reconnect the AirPods (put them in the case and take them out) before testing."
