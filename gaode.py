"""高德地图地理编码和驾车路线数据处理。"""

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().with_name(".env"))

AMAP_KEY = os.getenv("AMAP_KEY")
if not AMAP_KEY:
    raise RuntimeError("缺少环境变量 AMAP_KEY，请在项目 .env 文件中配置")
AMAP_URL = "https://restapi.amap.com/v3"


def geocode(address: str) -> str:
    """将地点转换为高德坐标，返回 ``经度,纬度``。

    标准地址优先使用地理编码；景区、酒店等地点在地址编码失败时，
    自动改用 POI 关键字搜索。
    """
    response = requests.get(
        f"{AMAP_URL}/geocode/geo",
        params={"address": address, "key": AMAP_KEY},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("status") == "1" and data.get("geocodes"):
        return data["geocodes"][0]["location"]

    geocode_error = data
    response = requests.get(
        f"{AMAP_URL}/place/text",
        params={
            "keywords": address,
            "key": AMAP_KEY,
            "offset": 10,
            "page": 1,
            "extensions": "base",
        },
        timeout=15,
    )
    response.raise_for_status()
    poi_data = response.json()
    pois = poi_data.get("pois", [])
    if poi_data.get("status") == "1" and pois and pois[0].get("location"):
        return pois[0]["location"]

    raise ValueError(
        f"地点解析失败：{address}，"
        f"地址编码返回：{geocode_error}，POI 搜索返回：{poi_data}"
    )


def geocode_candidates(
    address: str,
    limit: int = 5,
    reference_location: str | None = None,
) -> list[str]:
    """返回地点可能对应的多个坐标，用于解决同名景点定位歧义。"""
    candidates: list[str] = []
    exact_candidates: list[str] = []
    nearby_candidates: list[str] = []
    nearby_exact_candidates: list[str] = []

    def add_pois(items: list[dict[str, Any]], *, nearby: bool = False) -> None:
        target = "".join(address.split()).casefold()
        for item in items:
            location = item.get("location")
            if not location:
                continue
            destination = nearby_candidates if nearby else candidates
            exact_destination = (
                nearby_exact_candidates if nearby else exact_candidates
            )
            destination.append(location)
            name = "".join(str(item.get("name", "")).split()).casefold()
            if name == target:
                exact_destination.append(location)

    # 对行程中的下一站优先做周边搜索，可避免选到外省同名景点。
    if reference_location:
        response = requests.get(
            f"{AMAP_URL}/place/around",
            params={
                "location": reference_location,
                "keywords": address,
                "radius": 50000,
                "sortrule": "distance",
                "key": AMAP_KEY,
                "offset": limit,
                "page": 1,
                "extensions": "base",
            },
            timeout=15,
        )
        response.raise_for_status()
        around_data = response.json()
        if around_data.get("status") == "1":
            add_pois(around_data.get("pois", []), nearby=True)

    response = requests.get(
        f"{AMAP_URL}/geocode/geo",
        params={"address": address, "key": AMAP_KEY},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("status") == "1":
        candidates.extend(
            item["location"]
            for item in data.get("geocodes", [])
            if item.get("location")
        )

    response = requests.get(
        f"{AMAP_URL}/place/text",
        params={
            "keywords": address,
            "key": AMAP_KEY,
            "offset": max(limit, 10),
            "page": 1,
            "extensions": "base",
        },
        timeout=15,
    )
    response.raise_for_status()
    poi_data = response.json()
    if poi_data.get("status") == "1":
        add_pois(poi_data.get("pois", []))

    # 周边搜索的地理上下文比全国范围内的名称完全匹配更可靠。例如用户从
    # 额尔古纳湿地前往“白桦林景区”时，周边结果可能叫“额尔古纳白桦林景区”，
    # 而全国搜索存在名称完全相同的新疆 POI。旧逻辑会因此丢掉正确的周边结果。
    preferred = (
        nearby_exact_candidates
        or nearby_candidates
        or exact_candidates
        or candidates
    )
    unique_candidates = list(dict.fromkeys(preferred))[:limit]
    if not unique_candidates:
        raise ValueError(
            f"地点解析失败：{address}，地址编码返回：{data}，"
            f"POI 搜索返回：{poi_data}"
        )
    return unique_candidates


def driving_route(origin_loc: str, dest_loc: str) -> list[dict[str, Any]]:
    """查询驾车路线，坐标格式为 ``经度,纬度``。"""
    response = requests.get(
        f"{AMAP_URL}/direction/driving",
        params={
            "key": AMAP_KEY,
            "origin": origin_loc,
            "destination": dest_loc,
            "extensions": "all",
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "1" or "route" not in data:
        raise ValueError(f"路线查询失败，返回：{data}")
    return data["route"].get("paths", [])


def organize_routes(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """整理 ``route.paths``，保留路线摘要和可逐段使用的导航信息。

    返回的数据结构示例：
    ``[{"route_index": 1, "summary": {...}, "steps": [{...}]}]``。
    """
    organized = []
    for route_index, path in enumerate(paths, start=1):
        steps = []
        for step_index, step in enumerate(path.get("steps", []), start=1):
            steps.append(
                {
                    "step_index": step_index,
                    "instruction": step.get("instruction", ""),
                    "road": step.get("road", ""),
                    "distance_m": int(step.get("distance", 0) or 0),
                    "duration_s": int(step.get("duration", 0) or 0),
                    "polyline": step.get("polyline", ""),
                }
            )

        organized.append(
            {
                "route_index": route_index,
                "summary": {
                    "distance_m": int(path.get("distance", 0) or 0),
                    "duration_s": int(path.get("duration", 0) or 0),
                    "strategy": path.get("strategy", ""),
                    "tolls": path.get("tolls", "0"),
                    "toll_distance_m": int(path.get("toll_distance", 0) or 0),
                    "traffic_lights": int(path.get("traffic_lights", 0) or 0),
                },
                "steps": steps,
            }
        )
    return organized


if __name__ == "__main__":
    origin_loc = geocode("合作市")
    dest_loc = geocode("扎尕那")
    routes = organize_routes(driving_route(origin_loc, dest_loc))

    for route in routes:
        summary = route["summary"]
        print(
            f"路线 {route['route_index']}："
            f"{summary['distance_m'] / 1000:.1f} 公里，"
            f"约 {summary['duration_s'] / 3600:.1f} 小时，"
            f"收费 {summary['tolls']} 元"
        )
        for step in route["steps"]:
            print(f"  {step['step_index']}. {step['instruction']}")
