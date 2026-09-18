# RoboCurate 源码预览与许可状态

当前版本提供独立启动、合成演示、第三方许可说明和自动测试配置。目标仓库为 [Lee-hz/RoboCurate](https://github.com/Lee-hz/RoboCurate)，下载方式见 [README](README.zh-CN.md)。

完整开源许可审核仍有一项待确认：

最初提供的 `mcap_viz` 来源或再分发授权，涉及 [16 个继承或改写的文件](docs/licensing/base-viewer-files.json)。详见 [许可证状态](LICENSE_STATUS.md)。拟用 Apache-2.0 的文本目前仅作为提案保存，没有覆盖这些待核实文件。

已识别的 Three.js、Unitree G1 模型、WARP-RM、HFlow、trajlens、Font Awesome 资源保留各自来源和许可证，见 [第三方说明](THIRD_PARTY_NOTICES.md)。源码包不含原始录像、审核库、浏览器登录状态或模型权重。`docs/images/` 包含本次选定的三张真实界面截图和一张合成演示图；运行时可生成带明显标识的合成 MCAP 试用数据。

源码包已补入完整的项目训练、校准、消融及验证代码和冻结协议。现有 G1 v1/v2/v3、公开视频时序模型及 DINOv2 图像骨干已作为独立 Release 权重包发布，源码内含配套安装入口。私有训练录像、标签、特征及折内实验输出仍不公开，因此不能只靠下载包复算此前任务准确率。详见 [模型范围](docs/MODELS.md) 和 [训练说明](docs/TRAINING.md)。

机器可读状态位于 [RELEASE_STATUS.json](RELEASE_STATUS.json)，分别记录 GitHub 上传状态与完整开源许可审核状态。打包采用显式文件清单；发布包内 `SOURCE_MANIFEST.json` 记录相对路径、大小和 SHA-256。
