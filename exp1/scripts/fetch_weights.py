"""P0.5 基座权重导出脚本。

将 Qwen3-0.6B 的权重与分词器文件从 HuggingFace 缓存导出为普通文件目录，供上传
OSS 后在 DLC 容器内以挂载方式加载。导出只做复制与校验，不改变文件内容，故容器内
加载到的权重与官方源逐字节一致。

权重下载在容器内受镜像源带宽波动影响，实测耗时在 110.6 秒至 684.0 秒之间，见
`LOG/02-问题与解决.md` Q8。改为预置后，任务内不再访问网络。

本机访问官方源须经代理且不能设置 HF_ENDPOINT，见 `RECORD/02-环境信息.md` 第一节。
调用示例

    set HTTPS_PROXY=http://127.0.0.1:7897
    python scripts/fetch_weights.py --repo-id Qwen/Qwen3-0.6B --output-dir models/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


# 容器内 from_pretrained 所需的最小文件集，缺一即无法离线加载
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)

# 仅保留 safetensors 形态，排除 hub 上的 bin 与各类框架导出格式
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors(path: Path) -> tuple[list[str], int, bool]:
    """读出 safetensors 的头部，返回精度集合、张量数据字节数与长度自洽性。

    文件布局为前 8 字节的小端长度、紧随其后的 JSON 头部、再之后为张量数据区。
    三段长度若不自洽，说明文件在下载或上传中被截断。
    """
    with path.open("rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length))

    tensors = {name: meta for name, meta in header.items() if name != "__metadata__"}
    dtypes = sorted({meta["dtype"] for meta in tensors.values()})
    data_bytes = sum(meta["data_offsets"][1] - meta["data_offsets"][0] for meta in tensors.values())

    expected_total = 8 + header_length + data_bytes
    return dtypes, data_bytes, path.stat().st_size == expected_total


def main() -> int:
    parser = argparse.ArgumentParser(description="导出基座权重供上传 OSS")
    parser.add_argument("--repo-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", default="models/Qwen3-0.6B")
    parser.add_argument("--revision", default=None, help="留空取默认分支的最新提交")
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        print(f"检测到 HF_ENDPOINT 为 {endpoint}")
        print("该变量会把请求导向镜像源。按环境信息的约定，本机下载应经代理直连官方源，")
        print("请先清除该变量再运行本脚本。")
        return 1

    if not (os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")):
        print("未检测到 HTTPS_PROXY，本机直连官方源不可达，须显式指定代理。")
        print("请参照模块文档中的调用示例。")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n导出 {args.repo_id} 至 {output_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=output_dir,
        allow_patterns=ALLOW_PATTERNS,
    )

    missing = [name for name in REQUIRED_FILES if not (output_dir / name).is_file()]
    if missing:
        print(f"缺少必需文件 {missing}，无法用于容器内离线加载。")
        return 1

    manifest: list[dict[str, object]] = []
    total_bytes = 0
    print("\n文件清单")
    for name in REQUIRED_FILES:
        path = output_dir / name
        size = path.stat().st_size
        total_bytes += size
        checksum = sha256_of(path)
        print(f"  {name}  {size / 1024**2:.2f} MiB  {checksum[:16]}")
        manifest.append({"file": name, "bytes": size, "sha256": checksum})

    weights = output_dir / "model.safetensors"
    dtypes, data_bytes, consistent = inspect_safetensors(weights)
    print(f"\n权重精度 {dtypes}，张量数据 {data_bytes / 1024**2:.2f} MiB")
    print(f"长度自洽 {'是' if consistent else '否'}，与文件大小比对{'一致' if consistent else '不一致'}")

    print(f"\n合计 {total_bytes / 1024**3:.2f} GiB，共 {len(REQUIRED_FILES)} 个文件")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "revision": args.revision or "默认分支",
                "files": manifest,
                "total_bytes": total_bytes,
                "dtypes": dtypes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"清单已写出 {manifest_path}")

    if not consistent:
        print("\n权重文件长度不自洽，请删除目录后重新导出。")
        return 1

    print("\n上传指引")
    print(f"  逐文件上传至 OSS 前缀 exp1/models/{output_dir.name}/，与容器路径逐段对应")
    print("  上传后与本表逐项核对大小，控制台显示的字节数应与本表一致")
    print("  前缀末尾不加斜杠，以免产生空路径段")
    print("  manifest.json 不参与加载，随文件一并上传可留作核对依据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
