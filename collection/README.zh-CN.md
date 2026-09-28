# collection-hub / collection-node：采集设备管理与数据同步

[English](README.md)

管理录制数据的机器人，并把录制完成的 episode 拉取到运行 RoboCurate 的存储服务器。

```
采集设备 ×N    collection-node（用户级 systemd：Nice=10 / IO idle / CPUQuota=5% / MemoryMax=64M）
  ├ 每 1 s     看最近 episode 的 mcap/__RECORDING__ 标记和 MCAP 增长 → 录制开始/结束时立即心跳
  ├ 每 5–15 s  PICO episode 服务的 /status（仅作参考）
  ├ 每 30 s    磁盘剩余 + 增量清单（文件、大小、mtime、sidecar 摘要）
  └ 心跳       POST :8422/api/node/heartbeat（Bearer token）
存储服务器     collection-hub（独立进程，hub.sqlite3）
  ├ 在线 / 录制 / 异常判定、双端磁盘等级、告警（可选 webhook）
  ├ rsync 调度（SSH + 设备端 rrsync 只读）→ <mirror_root>/<设备>/<设备上的原始目录>
  ├ 逐文件校验（大小 + mtime）→ 变更游标 GET /api/changes
  └ 管理 API 127.0.0.1:8423（RoboCurate 通过 /api/hub/* 代理）
存储服务器     RoboCurate：设备管理页、全局磁盘横幅；只重扫刚同步的 session
```

node 只依赖 Python 标准库，安装即拷贝一个文件。hub 独立于 RoboCurate：重启审核工具不会中断监控和传输，RoboCurate 也不持有任何设备凭据。

## 规则

来自对 PICO episode 服务和 MCAP 录制器的生命周期测试（正常停止、停止后标注、删除、录制进程崩溃、录制中服务重启）。

| 判定 | 规则 |
|---|---|
| 正在录 | `__RECORDING__` 存在且 MCAP 在增长（开始后 15 s 宽限），或没有标记但 MCAP 仍在增长（服务重启后的孤儿录制）。`/status` 的 `recorder.active` 只用于显示和“状态不一致”告警：录制进程崩溃后它一直为 true，孤儿录制时它为 false |
| 录制中断 | 标记存在但 MCAP 超过 30 s 不增长 → 异常告警；10 分钟后按原样同步 |
| 可同步 | 有 `mcap/metadata.yaml`、无标记，且不再是当前 episode 或已静默 `settle_seconds`（默认 120 s，操作员仍可能删除当前这一条）；只留下 sidecar 的失败 episode 也同步 |
| 重传 | 文件指纹变化即重传（标注和元数据更新会改写 JSON）；MCAP 不变则不重复传输 |
| 设备上删除 | 已同步的副本移到 `<mirror_root>/.trash/`，不删除 |
| 注册前的历史数据 | 默认不同步，可按设备开启 |
| 同步时机 | 设备录制中不同步；停止 `idle_minutes`（默认 10）后开始，“收工”或“立即同步”跳过等待。任一设备录制时总速率限为 `busy_bwlimit_mb`（默认 15 MB/s，按任务平分），保护遥操作 Wi-Fi；否则全速。策略变化时中断并以 `--partial` 续传 |
| 容量 | 服务器剩余空间将低于 `reserve_gb` 时不发起新同步 |
| 可关机 | 在线、未录制、无等待标注、无转换进行中、无待同步/刚停止/写入未完成、无同步任务 |
| 时钟偏差 | 由心跳测得；设备时钟的 mtime 按设备时钟比较 |

## 安装

存储服务器（用户级服务，开启 linger 以便开机自启）：

```bash
collection/deploy/install_hub.sh --mirror-root /data/recordings/raw --reserve-gb 100
cp collection/deploy/robocurate.service ~/.config/systemd/user/   # 可选：RoboCurate 作为服务运行
```

在 RoboCurate 的 **设备管理 → 添加设备** 生成一次性令牌和设备端命令，形如：

```bash
curl -fsSL http://<hub-host>:8422/install/install_node.sh | bash -s -- \
  --hub http://<hub-host>:8422 --token <token> --hub-key 'ssh-ed25519 AAAA… collection-hub@<hub-host>'
```

命令会安装 `collection-node` 用户服务，并把 hub 公钥写入 `authorized_keys`，限制为 `rrsync -ro <数据目录>`。命令行方式：`python3 collection/hub.py --config ~/.config/collection-hub/config.json add-device --name G1-01 --ssh-target <user>@<robot-host>`。

可选能力（默认关闭）：

* `--allow-shutdown`：页面“同步完关机”。需要 sudoers：`<user> ALL=(root) NOPASSWD: /sbin/shutdown`。
* `--allow-cleanup`：页面“清理已同步数据”。hub 只用逐文件复核过的副本生成预览，需输入确认码；node 只删除文件未变、且不是当前或录制中的 episode。

## 配置（`~/.config/collection-hub/config.json`）

`mirror_root`、`reserve_gb`、`max_concurrent`、`busy_bwlimit_mb`、`idle_minutes`、`settle_seconds`、`batch_max_gb`、`batch_max_episodes`、`default_write_rate_gb_h`、`default_daily_intake_gb`（实测满 3 天前使用）、`device_warn_hours`/`device_critical_hours`、`server_warn_days`/`server_critical_days`、`webhook` 与 `webhook_format`（`feishu`|`generic`）、`node_listen`、`admin_listen`。

## 安全

node 端口（`:8422`）面向可信的采集局域网：心跳需要每台设备独立的令牌（只存哈希），安装文件公开。管理 API 只监听本机回环；RoboCurate 按自身同源规则转发，能打开 RoboCurate 的人也能控制同步，见 [SECURITY.md](../SECURITY.md)。node 上报的路径会校验，rsync 不复制符号链接和设备文件。

## 排查

```bash
systemctl --user status collection-hub;  journalctl --user -u collection-hub -f    # 服务器
systemctl --user status collection-node; journalctl --user -u collection-node -f   # 设备
python3 collection/hub.py --config ~/.config/collection-hub/config.json list
python3 ~/.local/share/collection-node/node.py --config ~/.config/collection-node/config.json --once
```

测试：`python -m pytest tests/test_collection.py`（真实 hub、node 和 rsync 本地传输，使用临时目录）。
