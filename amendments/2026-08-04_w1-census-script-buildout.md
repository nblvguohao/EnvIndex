# Amendment 2026-08-04: W1 数据普查脚本构建（t3_brapi_export + t3_login）

- **日期**：2026-08-04
- **修改范围**：protocol_freeze_paper2.md W1–W3 数据普查的数据获取工具；EnvIndex 仓库 `scripts/`
- **性质**：记录数据管道工具的开发与关键决策；协议正文不原位修改
- **代码提交**：EnvIndex 仓库 `7634d8f`（脚本+测试）、`69d9817`（登录助手）、`6c4eb1b`（重试）、`4727d8a`（并行 census + 分页修复）

---

## 1. 交付脚本

| 脚本 | 功能 |
|------|------|
| `scripts/t3_login.py` | T3 用户名/密码 → BrAPI token（POST `/brapi/v1/token`，~2h 有效期）；getpass 不回显；token 存 `data/t3/.t3_token`（gitignore） |
| `scripts/t3_brapi_export.py` | BrAPI V2 客户端：census 模式（枚举项目→试验→标记物候性状）与 export 模式（拉逐小区物候观测） |

配套测试 `tests/test_t3_brapi_export.py`（12 项，mock HTTP，零网络）。

## 2. 真实连通后的关键修正（超出 mock 测试可覆盖范围）

1. **认证**：T3 要求 bearer token（匿名 401）；`/brapi/v1/token` 端点用 username/password POST 换 token（v1 响应含 `access_token` 与 `expires_in`）。
2. **基址**：BrAPI v2 路径挂在 `<root>/brapi/v2/` 下；脚本曾打到无前缀的网页端点（返回 XML HTML）→ 客户端在 `__post_init__` 中归一化基址（含 `/brapi/v2/` 且不重复追加）。
3. **分页**：T3 把分页放在 `metadata.pagination`（而非 BrAPI v2 文档示例的 `result.pagination`）→ 脚本原先每个端点只取第 0 页；已修复为两者都探测。
4. **网络稳定性**：到 T3 的连接间歇性被重置/截断 → `_call_with_retry` 指数退避重试（连接重置/超时/截断 JSON/HTTP 429+5xx；401/403 立即失败）。实际运行时部分请求确实触发重试。
5. **单个试验 500**：个别试验的 `observationvariables` 端点返回 HTTP 500（非瞬时）→ census 优雅降级，记录 `variables_error` 继续，不中断整体。
6. **性能**：~1090 试验/3 个项目逐个拉性状列表串行需 >5 分钟 → 改为 `ThreadPoolExecutor` 并行（`--workers`，默认 6），配 `flush=True` 进度输出。

## 3. 实测数据点

- 项目总数：83；匹配 "Nebraska" 3 个（Nebraska USDA 362 试验 / University of Nebraska 349 / USDA-ARS Nebraska 379）。
- `pageSize=1000` 被服务器实际封顶（~83 项目/页，20750 字节 JSON 正常解析）。
- token 有效期 ~7200s（2h），到期需重新 `t3_login.py`。

## 4. 对已执行分析的影响

- 无已产出分析；本 amendment 仅记录数据获取工具。census 结果分析在 W1 评审时进行。
- 已知局限：census 逐试验拉 `observationvariables` 是主成本（即使并行）；全 83 项目普查预计需显著更长时间——W1 建议先按目标育种项目分批普查，而非全库扫描。
