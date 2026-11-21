"""下载教师公开分享的实验一数据；不持久化临时授权令牌。"""
import hashlib
import json
from pathlib import Path
import zipfile

import requests

ROOT = Path(__file__).resolve().parent
BASE = "https://bhpan.buaa.edu.cn"
LINK_ID = "AAB14093C4DF1F4085922390C07C87498F"
NAMES = ("1-Digit-TrainSet.zip", "1-Digit-TestSet.zip")


def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.get(f"{BASE}/link/{LINK_ID}", timeout=60).raise_for_status()
    token = next((c.value for c in session.cookies if c.name == f"link_token:{LINK_ID}"), None)
    if not token:
        raise RuntimeError("共享链接未返回匿名访问令牌；请按 README 手动下载数据。")
    session.headers["Authorization"] = f"Bearer {token}"
    response = session.get(f"{BASE}/api/efast/v1/entry-item", timeout=60)
    response.raise_for_status()
    root_id = response.json()[0]["id"]
    response = session.post(f"{BASE}/api/efast/v1/dir/list", json={"docid": root_id}, timeout=60)
    response.raise_for_status()
    files = {f["name"]: f for f in response.json()["files"]}
    provenance = {"source_url": f"{BASE}/link/{LINK_ID}", "files": {}}
    for name in NAMES:
        item = files[name]
        path = raw / name
        if not (path.exists() and path.stat().st_size == item["size"] and zipfile.is_zipfile(path)):
            response = session.post(f"{BASE}/api/efast/v1/file/osdownload", json={
                "docid": item["docid"], "authtype": "QUERY_STRING", "savename": name,
                "usehttps": True, "rev": item["rev"],
            }, timeout=60)
            response.raise_for_status()
            method, url = response.json()["authrequest"]
            if method.upper() != "GET":
                raise RuntimeError("服务器返回了不支持的下载方法")
            temp = path.with_suffix(".zip.part")
            print(f"下载 {name}: {item['size'] / 1e6:.1f} MB", flush=True)
            # 使用独立请求，避免把共享页的 Authorization 发送给对象存储。
            with requests.get(url, stream=True, timeout=(30, 120)) as stream:
                stream.raise_for_status()
                with temp.open("wb") as out:
                    for chunk in stream.iter_content(1 << 20):
                        out.write(chunk)
            if temp.stat().st_size != item["size"] or not zipfile.is_zipfile(temp):
                raise RuntimeError(f"{name} 下载不完整；请重新运行")
            with zipfile.ZipFile(temp) as archive:
                bad = archive.testzip()
                if bad:
                    raise RuntimeError(f"ZIP 校验失败: {bad}")
            temp.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        provenance["files"][name] = {"bytes": path.stat().st_size, "sha256": digest}
        print(f"已校验 {name}: SHA256={digest}", flush=True)
    (raw / "download_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
