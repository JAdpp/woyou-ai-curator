"""Build the frozen 250-question retrieval evaluation set from global_open.

The builder never calls an LLM or retrieval provider.  Field-addressable
questions receive exhaustive deterministic qrels over the frozen catalogue.
Open semantic and visual questions receive only metadata/evidence candidate
pools marked ``pooled_silver_pending_human_review`` with relevance 0; those
rows must never be interpreted as human relevance gold.

The generated files are committed as a frozen benchmark.  Re-running this
script is a reproducibility check, not permission to silently replace labels or
questions after observing system results.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECTION_DIR = PROJECT_ROOT / "data" / "collections" / "global_open"
OUTPUT_DIR = PROJECT_ROOT / "data" / "qa" / "retrieval_eval_v1"
FROZEN_AT = "2026-08-31T18:00:00+08:00"
BENCHMARK_VERSION = "20260831-v1"

CATEGORY_COUNTS = {
    "exact_title_author_institution": 35,
    "synonym_bilingual_typo": 30,
    "date_material_region": 35,
    "open_theme": 35,
    "cross_cultural": 35,
    "symbolic_causal_multihop": 30,
    "visual_motif": 25,
    "ambiguous_out_of_scope": 25,
}

PACK_LABELS = {
    "east_asia": "东亚",
    "south_asia": "南亚",
    "southeast_asia": "东南亚",
    "west_asia_north_africa": "西亚与北非",
    "europe": "欧洲",
    "africa": "非洲",
    "americas": "美洲",
    "oceania": "大洋洲",
}

# These anchors identify examples repeatedly used during earlier debugging.
# Exact comparison against every earlier QA question is performed separately.
LEGACY_BANNED_PATTERNS = [
    r"猫|\bcat(?:s)?\b",
    r"狗狗|狗在|犬类|\bdog(?:s)?\b|\bcanine\b",
    r"相似的蓝|蓝色|\bcobalt blue\b",
    r"皇帝|王权|统治者|\bemperor\b|\bruler(?:s)?\b",
    r"大象|象牙|\belephant(?:s)?\b|\bivory\b",
    r"亲子|母子|\bmother and child\b|\bparent[- ]child\b",
    r"漆器|\blacquer\b",
    r"贝壳.*镶|\bshell inlay\b",
    r"梳妆",
    r"朝圣",
    r"人类遗骸",
    r"量子",
    r"饮酒|酒具",
    r"棋盘|骰子|纸牌",
    r"风怎么被看见",
    r"湖.*地图",
    r"接布|翻面.*缝线",
    r"工具.*磨",
    r"机器.*声音",
    r"能乐|Gẹ̀lẹ́dé|Yup'ik",
    r"没有签名",
    r"扫地机器人",
    r"DNA|双螺旋",
    r"青铜器.*生锈",
    r"宗教器物.*神圣",
    r"羽蛇神",
    r"拍卖价格",
    r"治疗焦虑",
    r"假文物",
    r"所有文明.*崇拜太阳",
]
LEGACY_BANNED_RE = [re.compile(pattern, re.IGNORECASE) for pattern in LEGACY_BANNED_PATTERNS]

MATERIAL_ALIASES = [
    ("gold", "黄金"),
    ("silver", "银"),
    ("iron", "铁"),
    ("steel", "钢"),
    ("copper", "铜"),
    ("wood", "木"),
    ("silk", "丝"),
    ("paper", "纸"),
    ("glass", "玻璃"),
    ("porcelain", "瓷"),
    ("terracotta", "陶土"),
    ("marble", "大理石"),
    ("linen", "亚麻"),
    ("leather", "皮革"),
    ("graphite", "石墨"),
    ("watercolor", "水彩"),
    ("canvas", "画布"),
    ("jade", "玉"),
    ("enamel", "珐琅"),
    ("ceramic", "陶瓷"),
]

TYPE_ALIASES = [
    ("painting", "绘画"),
    ("print", "版画"),
    ("sculpture", "雕塑"),
    ("ceramic", "陶瓷器"),
    ("metalwork", "金属工艺"),
    ("photograph", "摄影作品"),
    ("jewelry", "首饰"),
    ("drawing", "素描"),
    ("textile", "纺织品"),
    ("manuscript", "手稿"),
    ("glass", "玻璃器"),
    ("coins", "钱币"),
    ("basketry", "编篮"),
    ("furniture and woodwork", "家具与木作"),
    ("musical instrument", "乐器"),
]

OPEN_THEME_SPECS = [
    ("窗、帘与屏风怎样在室内图像中安排可见与不可见？", ["window", "curtain", "screen"]),
    ("灯、烛与暗部怎样把夜晚变成可以观看的空间？", ["lamp", "candle", "night"]),
    ("桥梁在作品里何时是交通设施，何时又成为空间分界？", ["bridge", "river", "road"]),
    ("船只图像怎样把旅行、劳动与危险放进同一条观看线索？", ["boat", "ship", "voyage"]),
    ("雨、雪与雾在馆藏记录中分别通过哪些形状和材料被表现？", ["rain", "snow", "mist", "fog"]),
    ("残片与废墟怎样让缺失本身成为作品意义的一部分？", ["fragment", "ruin", "broken"]),
    ("钥匙、锁与门闩怎样把安全、私密和控制变成可见的物件？", ["key", "lock", "door"]),
    ("塔楼与高处视点怎样改变作品对城市和地景的组织？", ["tower", "city", "view"]),
    ("厨房器具能否让我们看到备餐、加热与分享食物的不同环节？", ["kitchen", "cooking", "food", "pot"]),
    ("鞋、帽和手套怎样在保护身体之外传递职业与场合信息？", ["shoe", "hat", "glove", "costume"]),
    ("乐器的形制和材料记录能告诉我们哪些演奏动作？", ["musical instrument", "music", "string", "drum"]),
    ("书信、信封与书写工具怎样留下远距离交流的痕迹？", ["letter", "correspondence", "writing"]),
    ("钱币上的人物、文字与边缘装饰怎样共同制造可信度？", ["coin", "inscription", "portrait"]),
    ("钟、表与计时装置怎样把时间从经验变成可以携带的刻度？", ["clock", "watch", "time"]),
    ("首饰的连接、扣合与佩戴结构怎样影响它被观看的方式？", ["jewelry", "necklace", "bracelet", "brooch"]),
    ("武器上的装饰何时会遮蔽它原本的使用功能？", ["weapon", "sword", "armor", "decoration"]),
    ("桌、椅与柜的尺寸和构造怎样提示它们原来的使用场景？", ["table", "chair", "cabinet", "furniture"]),
    ("肖像背景里的书、帷幕和建筑细节能补充哪些人物信息？", ["portrait", "book", "curtain", "architecture"]),
    ("重复的边框和带状装饰怎样引导眼睛沿着对象表面移动？", ["border", "band", "repeat", "ornament"]),
    ("画面中的空白何时是未完成，何时是主动保留的空间？", ["unfinished", "blank", "negative space"]),
    ("手部动作在肖像和叙事图像中怎样提示交流、劳动或拒绝？", ["hand", "gesture", "portrait"]),
    ("鸟的飞行、停栖与成群状态怎样改变图像的节奏？", ["bird", "flight", "wing"]),
    ("鱼和水生生物怎样把水下环境带到器物表面？", ["fish", "aquatic", "water"]),
    ("蛇形曲线怎样跨越首饰、器皿与图像边框？", ["snake", "serpent", "curve"]),
    ("狮子的鬃毛、爪与正面凝视怎样被不同媒介简化？", ["lion", "mane", "claw"]),
    ("马的奔跑、站立与负载状态怎样提示不同的叙事时刻？", ["horse", "rider", "equestrian"]),
    ("海岸、港口与开阔水面怎样组织远近关系？", ["coast", "harbor", "sea"]),
    ("石材表面的凿痕、抛光与风化怎样共同记录时间？", ["stone", "carving", "polish", "weathered"]),
    ("玻璃的透明、反光与着色怎样改变对象内部和外部的边界？", ["glass", "transparent", "reflection"]),
    ("纸张的折叠、卷起与装订怎样决定观看顺序？", ["paper", "fold", "scroll", "binding"]),
    ("线、绳与结怎样从结构部件变成表面装饰？", ["thread", "rope", "knot", "cord"]),
    ("微型对象怎样利用尺度让观看者靠得更近？", ["miniature", "small", "scale"]),
    ("框、底座与悬挂装置怎样改变对象被当成作品观看的方式？", ["frame", "mount", "hanging"]),
    ("火焰、烟与余烬怎样在静止图像里暗示正在发生的变化？", ["flame", "fire", "smoke"]),
    ("道路、台阶与坡面怎样在平面图像中形成行进方向？", ["road", "stairs", "path"]),
    ("书页的边注、删改与补写怎样让阅读过程留下可见痕迹？", ["marginalia", "annotation", "manuscript"]),
    ("容器的盖、口沿与把手怎样提示开启、倾倒和携带的动作？", ["lid", "handle", "rim", "vessel"]),
    ("建筑模型和微缩景观怎样压缩真实空间的复杂关系？", ["model", "miniature", "architecture"]),
    ("阴影在摄影、素描和版画中分别承担轮廓还是空间深度？", ["shadow", "photograph", "drawing", "print"]),
    ("纤维的松紧、密度与起伏怎样在图像之外形成触觉线索？", ["fiber", "weaving", "texture"]),
]

CROSS_CULTURAL_TOPICS = [
    ("绘画的基底与颜料", ["painting", "paint"]),
    ("版画的印制痕迹", ["print", "printing"]),
    ("雕塑的体量与表面", ["sculpture", "carving"]),
    ("陶瓷对象的器壁与釉面", ["ceramic", "glaze"]),
    ("金属工艺的连接与装饰", ["metalwork", "metal"]),
    ("摄影对象的成像工艺", ["photograph", "photographic"]),
    ("首饰的佩戴与连接", ["jewelry", "ornament"]),
    ("素描的线条与擦改", ["drawing", "graphite"]),
    ("纺织品的组织结构", ["textile", "weaving"]),
    ("手稿的装订与阅读顺序", ["manuscript", "binding"]),
    ("玻璃对象的透明与着色", ["glass", "transparent"]),
    ("钱币的文字与边缘", ["coin", "inscription"]),
    ("编织容器的纤维结构", ["basket", "fiber"]),
    ("家具的承托与收纳结构", ["furniture", "wood"]),
    ("乐器的共鸣结构", ["musical instrument", "music"]),
    ("钟铃的铸造与悬挂", ["bell", "cast"]),
    ("纸本对象的折叠与展开", ["paper", "fold"]),
    ("木作对象的雕刻与拼接", ["wood", "carving"]),
    ("银质对象的锤揲与抛光", ["silver", "metal"]),
    ("黄金装饰的贴附与镶嵌", ["gold", "gilt"]),
    ("铁质对象的锻造与连接", ["iron", "forged"]),
    ("丝织对象的经纬与光泽", ["silk", "weaving"]),
    ("皮革对象的缝合与成形", ["leather", "stitched"]),
    ("瓷质对象的釉与烧成记录", ["porcelain", "glaze"]),
    ("陶土对象的塑形与烧制", ["terracotta", "fired"]),
    ("大理石对象的凿刻与抛光", ["marble", "carved"]),
    ("刺绣对象的线迹与底布", ["embroidery", "thread"]),
    ("蕾丝对象的孔隙与边缘", ["lace", "thread"]),
    ("护具的防护结构与表面装饰", ["armor", "helmet"]),
    ("工具的握持部位与工作端", ["tool", "handle"]),
    ("建筑构件的承接与装饰", ["architectural", "architecture"]),
    ("肖像中的服装与背景", ["portrait", "costume"]),
    ("风景图像的地平线与路径", ["landscape", "path"]),
    ("书籍对象的封面与书脊", ["book", "binding"]),
    ("灯具的燃料容器与出光口", ["lamp", "light"]),
    ("鞋履的鞋底与扣合", ["shoe", "costume"]),
    ("刀剑的刃、柄与鞘", ["sword", "blade"]),
    ("地图的比例、文字与边界", ["map", "border"]),
    ("圆盘状对象的中心与边缘", ["disc", "round"]),
    ("盒匣的盖合与内部空间", ["box", "lid"]),
    ("织毯的边缘与中心图案", ["carpet", "textile"]),
    ("印章的握持与压印面", ["seal", "impression"]),
    ("浮雕的前景与底面", ["relief", "carving"]),
    ("扇具的开合与支撑结构", ["fan", "folding"]),
    ("摄影相册的编排与装订", ["album", "photograph"]),
]

MULTIHOP_SPECS = [
    ("钥匙和锁的材料、开合结构与使用场景之间可能形成怎样的证据链？", ["key", "lock", "door"]),
    ("鞋底的材料和磨耗位置能否连接到行走环境与制作选择？", ["shoe", "sole", "leather"]),
    ("灯具的燃料容器、导热材料与摆放方式如何共同影响使用？", ["lamp", "oil", "metal"]),
    ("纸张的重量、折叠方式与装订形式怎样共同影响携带和阅读？", ["paper", "fold", "binding"]),
    ("玻璃的透明度、壁厚与展示位置之间能建立哪些可核查联系？", ["glass", "transparent", "thick"]),
    ("银器的反光、抛光工艺与陈设方式之间是否存在可追溯关系？", ["silver", "polish", "display"]),
    ("刀剑的材料、重量分配与装饰位置如何区分使用部件和展示部件？", ["sword", "blade", "decoration"]),
    ("乐器的材料、空腔结构与演奏姿势如何共同约束发声方式？", ["musical instrument", "sound", "wood"]),
    ("钱币的金属、压印工艺与流通磨损能否串联出使用过程？", ["coin", "metal", "worn"]),
    ("纺织品的纤维、织法与贸易来源如何共同影响成品尺寸？", ["textile", "fiber", "weaving"]),
    ("船体材料、连接方法与航行水域之间有哪些可以由记录支撑的关系？", ["boat", "ship", "wood"]),
    ("桥梁图像的结构形式、所在地点与纪念功能之间如何逐层求证？", ["bridge", "architecture", "memorial"]),
    ("书籍的纸张、书脊和批注如何连接制作、使用与保存三个阶段？", ["book", "binding", "annotation"]),
    ("肖像中的衣着、手持物与室内陈设能否共同支持人物角色判断？", ["portrait", "costume", "interior"]),
    ("摄影的底片工艺、复制数量与相册编排怎样共同改变图像传播？", ["photograph", "negative", "album"]),
    ("首饰的原料、扣合方式与磨损位置如何连接制作和佩戴过程？", ["jewelry", "clasp", "worn"]),
    ("护具的金属层、衬里与活动关节如何共同平衡防护和动作？", ["armor", "metal", "joint"]),
    ("钟铃的合金、壁厚和悬挂方式怎样共同约束振动？", ["bell", "metal", "hanging"]),
    ("陶瓷对象的胎体、釉层与烧成痕迹如何串联制作步骤？", ["ceramic", "glaze", "fired"]),
    ("石雕的石种、工具痕迹与安装位置能否共同说明施工次序？", ["stone", "carving", "architectural"]),
    ("木雕的木种、接缝和表面涂层如何连接结构与装饰？", ["wood", "carving", "coating"]),
    ("刺绣的底布、线材与针法怎样共同决定图案边缘？", ["embroidery", "thread", "stitch"]),
    ("地图的测绘信息、比例与出版方式如何共同影响边界表达？", ["map", "scale", "published"]),
    ("建筑残片的原始位置、断面与入藏记录如何共同支持复原假设？", ["architectural fragment", "fragment", "provenance"]),
    ("微型对象的尺寸、携带痕迹与收纳配件能否共同说明观看距离？", ["miniature", "case", "small"]),
    ("画框的材料、挂装方式与作品尺寸如何共同改变展示重心？", ["frame", "hanging", "painting"]),
    ("复制品的模具、材料差异与编号记录如何连接生产批次？", ["mold", "copy", "cast"]),
    ("织毯的纤维、染色与边缘损耗如何共同记录铺设和使用？", ["carpet", "fiber", "worn"]),
    ("印章的材质、刻面与印痕如何逐步连接到实际使用者？", ["seal", "inscription", "impression"]),
    ("器物上的把手、重心与容量信息如何共同提示倾倒动作？", ["handle", "vessel", "pour"]),
    ("浮雕的层次、观看高度与建筑位置如何共同形成叙事顺序？", ["relief", "architectural", "narrative"]),
    ("相册的装订、页序与题注如何共同改变单张照片的语境？", ["album", "photograph", "inscription"]),
]

VISUAL_SPECS = [
    ("同心圆和圆盘怎样在画面中制造中心与外围？", ["circle", "round", "disc"]),
    ("菱形网格怎样在纺织品和金属表面形成连续结构？", ["diamond", "lattice", "grid"]),
    ("平行条带怎样把对象表面分成不同阅读区？", ["stripe", "band", "border"]),
    ("棋格式明暗块怎样改变图案的方向感？", ["checker", "checkered", "square"]),
    ("螺旋线怎样连接中心、边缘与连续运动？", ["spiral", "scrollwork", "coil"]),
    ("波浪线怎样在不同材料上提示水面或运动？", ["wave", "water", "undulating"]),
    ("云团轮廓怎样遮挡、分隔或连接画面中的对象？", ["cloud", "sky"]),
    ("火焰形轮廓怎样通过尖端和重复制造上升感？", ["flame", "fire"]),
    ("翅膀的展开和收拢怎样改变形象的方向？", ["wing", "bird", "flight"]),
    ("眼睛的正面、侧面与重复排列怎样改变被注视感？", ["eye", "gaze", "face"]),
    ("手的张开、握持与指向怎样构成动作线索？", ["hand", "gesture"]),
    ("马的腿部排列怎样在静止图像中表现速度？", ["horse", "equestrian", "rider"]),
    ("狮子的正面脸与侧身轮廓怎样分配视觉重量？", ["lion", "face", "profile"]),
    ("蛇的盘绕与伸展怎样适应狭长或环形表面？", ["snake", "serpent", "coil"]),
    ("鱼鳞和鱼鳍的重复怎样形成方向一致的纹理？", ["fish", "scale", "fin"]),
    ("船帆与船身的三角和横向结构怎样形成平衡？", ["ship", "boat", "sail"]),
    ("塔的垂直线怎样与周围低矮形状形成对比？", ["tower", "vertical", "architecture"]),
    ("桥拱的重复怎样把两个岸边组织成连续画面？", ["bridge", "arch", "river"]),
    ("车轮和辐条怎样建立旋转中心？", ["wheel", "spoke", "vehicle"]),
    ("拱形开口怎样在平面中暗示更深的空间？", ["arch", "doorway", "architecture"]),
    ("柱列的间距怎样形成可数的视觉节拍？", ["column", "architecture", "row"]),
    ("山峰的重叠轮廓怎样建立前后层次？", ["mountain", "landscape"]),
    ("河流的弯曲线怎样带领视线穿过画面？", ["river", "landscape", "water"]),
    ("树枝的分叉怎样把画面切成大小不同的空隙？", ["tree", "branch"]),
    ("成群鸟形怎样通过疏密变化形成节奏？", ["bird", "flight", "group"]),
    ("阶梯形轮廓怎样从底部向上压缩空间？", ["stairs", "step", "architecture"]),
    ("椭圆形轮廓怎样区分正视和倾斜的圆面？", ["oval", "round", "vessel"]),
    ("交叉线怎样形成结点、网眼与方向变化？", ["cross", "line", "network"]),
]

AMBIGUOUS_SPECS = [
    ("给我找一些看起来很克制、但细看很复杂的东西。", ["minimal", "detail", "pattern"]),
    ("我想看有距离感的作品，但不知道该从哪种文化开始。", ["distance", "landscape", "view"]),
    ("有没有像在等待什么的物件？", ["waiting", "still", "figure"]),
    ("想看不那么宏大、但能慢慢看的东西。", ["small", "miniature", "detail"]),
    ("找几件表面很安静、结构却紧张的作品。", ["structure", "line", "contrast"]),
    ("我只记得画面像一扇开着的空间。", ["window", "doorway", "interior"]),
    ("想看让人觉得轻、但材料其实很重的东西。", ["stone", "metal", "delicate"]),
    ("有没有边缘比中心更值得看的作品？", ["border", "edge", "frame"]),
    ("我想从“遮住”而不是“展示”开始逛。", ["cover", "veil", "screen"]),
    ("找几件会让我注意到尺度错觉的东西。", ["scale", "miniature", "monumental"]),
    ("不按年代，给我一条从粗糙到精细的观看线索。", ["rough", "polished", "texture"]),
    ("我说不上主题，只想看彼此差得很远却能放在一起的对象。", ["contrast", "different", "object"]),
]

OUT_OF_SCOPE_QUESTIONS = [
    "仅凭馆藏照片能否计算展柜在七级地震中的玻璃应力？",
    "能否从图片判断某件金属对象是否残留可接触的有毒化学物？",
    "请根据对象照片给出可直接用于复制承重构件的工程尺寸。",
    "能否从一张肖像照片识别真实人物并核验其身份证号码？",
    "能否从馆藏图像反推出未公开遗址的精确 GPS 坐标？",
    "请从图片估算对象内部放射性并确认可以徒手接触。",
    "能否根据纹样判断我的祖先属于哪个民族并出具证明？",
    "请为一件古代乐器写出保证还原原始音高的调音参数。",
    "能否根据照片确定宝石的实验室成分与鉴定等级？",
    "请预测未来一周对象所在城市的天气并安排运输航班。",
    "能否从照片计算木材剩余承载寿命并批准投入建筑使用？",
    "请恢复一件新媒体作品已经丢失的加密档案密码。",
    "能否只凭展品图片确认其内部虫卵已全部失活并签发检疫证明？",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _date_range_label(start: int, end: int) -> str:
    if end < 0:
        return f"公元前 {abs(start)}—{abs(end)} 年"
    if start < 0:
        return f"公元前 {abs(start)} 年至公元 {end} 年"
    return f"{start}—{end} 年"


def _safe_text(value: Any, *, minimum: int = 2, maximum: int = 100) -> bool:
    text = _normalise(value)
    return (
        minimum <= len(text) <= maximum
        and "\ufffd" not in text
        and not any(pattern.search(text) for pattern in LEGACY_BANNED_RE)
        and bool(re.search(r"[\w\u3400-\u9fff]", text))
    )


def _stable(items: Iterable[Any], salt: str, key=lambda item: item["id"]) -> list[Any]:
    return sorted(
        items,
        key=lambda item: hashlib.sha256(
            f"{salt}:{key(item)}".encode("utf-8")
        ).hexdigest(),
    )


def _evidence_ids(obj: dict[str, Any], *, matching_terms: Sequence[str] = ()) -> list[str]:
    evidence = [
        chunk
        for chunk in obj.get("evidence", [])
        if isinstance(chunk, dict)
        and isinstance(chunk.get("id"), str)
        and chunk.get("sourceKind") != "institution_provenance"
    ]
    if matching_terms:
        matching = [
            chunk
            for chunk in evidence
            if any(term.casefold() in _normalise(chunk.get("text")).casefold() for term in matching_terms)
        ]
        if matching:
            evidence = matching
    return [str(chunk["id"]) for chunk in evidence[:2]]


def _object_blob(obj: dict[str, Any]) -> dict[str, str]:
    tags = " ".join(str(value) for value in obj.get("tags", []) if isinstance(value, str))
    themes = " ".join(str(value) for value in obj.get("themes", []) if isinstance(value, str))
    evidence = " ".join(
        _normalise(chunk.get("text"))
        for chunk in obj.get("evidence", [])
        if isinstance(chunk, dict) and chunk.get("sourceKind") != "institution_provenance"
    )
    return {
        "title": _normalise(obj.get("title")).casefold(),
        "type": _normalise(obj.get("type")).casefold(),
        "material": _normalise(obj.get("material") or obj.get("medium")).casefold(),
        "tags": tags.casefold(),
        "themes": themes.casefold(),
        "description": (
            _normalise(obj.get("description")) + " " + _normalise(obj.get("altText"))
        ).casefold(),
        "evidence": evidence.casefold(),
    }


def _pool_candidates(
    objects: list[dict[str, Any]],
    blobs: dict[str, dict[str, str]],
    terms: Sequence[str],
    *,
    required_legs: Sequence[str] = (),
    limit: int = 12,
) -> list[tuple[dict[str, Any], float, list[str]]]:
    scored: list[tuple[dict[str, Any], float, list[str]]] = []
    for obj in objects:
        blob = blobs[obj["id"]]
        matched: list[str] = []
        score = 0.0
        for term in terms:
            needle = term.casefold()
            term_score = 0.0
            if needle in blob["title"]:
                term_score = max(term_score, 5.0)
            if needle in blob["type"]:
                term_score = max(term_score, 4.0)
            if needle in blob["tags"]:
                term_score = max(term_score, 3.5)
            if needle in blob["material"]:
                term_score = max(term_score, 3.0)
            if needle in blob["themes"]:
                term_score = max(term_score, 2.0)
            if needle in blob["description"]:
                term_score = max(term_score, 1.5)
            if needle in blob["evidence"]:
                term_score = max(term_score, 1.0)
            if term_score:
                score += term_score
                matched.append(term)
        if matched and _evidence_ids(obj, matching_terms=matched):
            scored.append((obj, score + 0.05 * len(matched), matched))
    scored.sort(key=lambda row: (-row[1], str(row[0]["id"])))

    return _select_candidates(scored, required_legs=required_legs, limit=limit)


def _select_candidates(
    scored: Sequence[tuple[dict[str, Any], float, list[str]]],
    *,
    required_legs: Sequence[str] = (),
    limit: int = 12,
) -> list[tuple[dict[str, Any], float, list[str]]]:

    selected: list[tuple[dict[str, Any], float, list[str]]] = []
    seen: set[str] = set()
    for leg in required_legs:
        leg_rows = [row for row in scored if leg in row[0].get("culturePackIds", [])]
        for row in leg_rows[:2]:
            if row[0]["id"] not in seen:
                selected.append(row)
                seen.add(row[0]["id"])
    for row in scored:
        if len(selected) >= limit:
            break
        if row[0]["id"] not in seen:
            selected.append(row)
            seen.add(row[0]["id"])
    return selected[:limit]


def _legacy_questions() -> set[str]:
    questions: set[str] = set()
    for path in (PROJECT_ROOT / "data" / "qa").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("cases", []):
            if isinstance(row, dict) and row.get("question"):
                questions.add(_normalise(row["question"]).casefold())
    for name in ("regression_questions.json", "question_cards.json"):
        path = COLLECTION_DIR / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("questions", payload.get("cards", payload)) if isinstance(payload, dict) else payload
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, str):
                    questions.add(_normalise(row).casefold())
                elif isinstance(row, dict) and row.get("question"):
                    questions.add(_normalise(row["question"]).casefold())
    return questions


def _gold_qrel(
    query_id: str,
    obj: dict[str, Any],
    *,
    rule: str,
    matched_fields: Sequence[str],
) -> dict[str, Any]:
    return {
        "queryId": query_id,
        "objectId": obj["id"],
        "relevance": 3,
        "supportingEvidenceIds": _evidence_ids(obj),
        "culturalLegs": list(obj.get("culturePackIds", [])),
        "judgmentStatus": "deterministic_field_gold",
        "provenance": "exact match over frozen institution metadata fields",
        "deterministicRule": rule,
        "matchedFields": list(matched_fields),
    }


def _pooled_qrel(
    query_id: str,
    row: tuple[dict[str, Any], float, list[str]],
) -> dict[str, Any]:
    obj, score, matched = row
    return {
        "queryId": query_id,
        "objectId": obj["id"],
        "relevance": 0,
        "supportingEvidenceIds": _evidence_ids(obj, matching_terms=matched),
        "culturalLegs": list(obj.get("culturePackIds", [])),
        "judgmentStatus": "pooled_silver_pending_human_review",
        "provenance": "deterministic metadata/evidence token pool; not human gold",
        "candidateScore": round(score, 4),
        "poolSources": ["title", "type", "material", "tags", "themes", "description", "evidence"],
        "matchedPoolTerms": matched,
    }


def build_dataset() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    collection_manifest = json.loads((COLLECTION_DIR / "manifest.json").read_text(encoding="utf-8"))
    objects_path = COLLECTION_DIR / "objects.json"
    if _sha256(objects_path) != str(collection_manifest["objectsSha256"]).casefold():
        raise RuntimeError("global_open objects.json does not match its manifest hash")
    payload = json.loads(objects_path.read_text(encoding="utf-8"))
    objects = payload.get("objects", payload) if isinstance(payload, dict) else payload
    if len(objects) != collection_manifest["objectCount"]:
        raise RuntimeError("global_open object count does not match its manifest")
    objects = [obj for obj in objects if _evidence_ids(obj)]
    blobs = {str(obj["id"]): _object_blob(obj) for obj in objects}
    legacy = _legacy_questions()
    pool_cache: dict[tuple[str, ...], list[tuple[dict[str, Any], float, list[str]]]] = {}

    def scored_pool(terms: Sequence[str]) -> list[tuple[dict[str, Any], float, list[str]]]:
        key = tuple(term.casefold() for term in terms)
        if key not in pool_cache:
            pool_cache[key] = _pool_candidates(
                objects,
                blobs,
                terms,
                limit=len(objects),
            )
        return pool_cache[key]

    questions: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    category_indices: Counter[str] = Counter()

    def add_question(
        category: str,
        text: str,
        *,
        judgment_mode: str,
        expected: str | None = None,
        required_legs: Sequence[str] = (),
        rule: dict[str, Any] | None = None,
        pool_terms: Sequence[str] = (),
        rows: Sequence[dict[str, Any]] = (),
    ) -> str:
        normalised = _normalise(text).casefold()
        if normalised in seen_questions or normalised in legacy:
            raise RuntimeError(f"duplicate or legacy question: {text}")
        if any(pattern.search(text) for pattern in LEGACY_BANNED_RE):
            raise RuntimeError(f"legacy theme leaked into question: {text}")
        seen_questions.add(normalised)
        category_indices[category] += 1
        query_id = f"{category_indices[category]:03d}-{category.replace('_', '-')}"
        question: dict[str, Any] = {
            "queryId": query_id,
            "category": category,
            "question": _normalise(text),
            "language": "zh-CN",
            "judgmentMode": judgment_mode,
            "freezeStatus": "frozen_before_system_comparison",
        }
        if expected:
            question["expectedAnswerability"] = expected
        if required_legs:
            question["requiredCulturalLegs"] = list(required_legs)
        if rule:
            question["deterministicRule"] = rule
        if pool_terms:
            question["poolQueryTerms"] = list(pool_terms)
        questions.append(question)
        for row in rows:
            qrels.append({**row, "queryId": query_id})
        return query_id

    # 1) Exact title / author / institution and accession lookups: 35.
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_maker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        title = _normalise(obj.get("title"))
        maker = _normalise(obj.get("maker") or obj.get("creator"))
        if _safe_text(title, minimum=8, maximum=85):
            by_title[title.casefold()].append(obj)
        if (
            _safe_text(maker, minimum=4, maximum=60)
            and "unknown" not in maker.casefold()
            and "anonymous" not in maker.casefold()
        ):
            by_maker[maker.casefold()].append(obj)

    title_groups = [rows for rows in by_title.values() if 1 <= len(rows) <= 6]
    for rows in _stable(title_groups, "exact-title", key=lambda group: group[0]["title"])[:12]:
        title = _normalise(rows[0]["title"])
        text = f"请定位馆藏题名恰好为《{title}》的全部记录，并核对各自的机构与年代。"
        query_id = add_question(
            "exact_title_author_institution",
            text,
            judgment_mode="deterministic_field_gold",
            expected="supported",
            rule={"field": "title", "operator": "casefold_exact", "value": title},
        )
        qrels.extend(
            _gold_qrel(query_id, obj, rule="title casefold exact", matched_fields=["title"])
            for obj in rows
        )

    maker_groups = [rows for rows in by_maker.values() if 1 <= len(rows) <= 6]
    for rows in _stable(maker_groups, "exact-maker", key=lambda group: group[0].get("maker") or group[0].get("creator"))[:12]:
        maker = _normalise(rows[0].get("maker") or rows[0].get("creator"))
        text = f"哪些馆藏把“{maker}”明确记录为 maker 或 creator？请返回规范题名与机构。"
        query_id = add_question(
            "exact_title_author_institution",
            text,
            judgment_mode="deterministic_field_gold",
            expected="supported",
            rule={"fields": ["maker", "creator"], "operator": "casefold_exact", "value": maker},
        )
        qrels.extend(
            _gold_qrel(query_id, obj, rule="maker/creator casefold exact", matched_fields=["maker", "creator"])
            for obj in rows
        )

    accession_candidates = [
        obj
        for obj in objects
        if _safe_text(obj.get("accessionNumber"), maximum=45)
        and _safe_text(obj.get("institution"), maximum=80)
        and _safe_text(obj.get("title"), minimum=5, maximum=90)
    ]
    for obj in _stable(accession_candidates, "exact-accession")[:11]:
        institution = _normalise(obj["institution"])
        accession = _normalise(obj["accessionNumber"])
        text = f"请定位 {institution} 馆藏号“{accession}”的对象，并核对题名、年代和材质。"
        query_id = add_question(
            "exact_title_author_institution",
            text,
            judgment_mode="deterministic_field_gold",
            expected="supported",
            rule={"fields": ["institution", "accessionNumber"], "operator": "casefold_exact_pair", "values": [institution, accession]},
        )
        qrels.append(
            _gold_qrel(query_id, obj, rule="institution + accession exact", matched_fields=["institution", "accessionNumber"])
        )

    # 2) Typos, Chinese/English type aliases and material aliases: 30.
    unique_title_objects = [rows[0] for rows in title_groups if len(rows) == 1]

    def typo_title(title: str) -> str | None:
        matches = [match for match in re.finditer(r"[A-Za-z]{6,}", title)]
        if not matches:
            return None
        match = max(matches, key=lambda item: len(item.group(0)))
        word = match.group(0)
        index = max(1, len(word) // 2 - 1)
        if word[index].casefold() == word[index + 1].casefold():
            index = 1
        mutated = word[:index] + word[index + 1] + word[index] + word[index + 2 :]
        return title[: match.start()] + mutated + title[match.end() :]

    typo_rows = []
    for obj in _stable(unique_title_objects, "typo-title"):
        typo = typo_title(_normalise(obj["title"]))
        if typo and typo != obj["title"] and _safe_text(typo, minimum=8, maximum=90):
            typo_rows.append((obj, typo))
        if len(typo_rows) == 10:
            break
    for obj, typo in typo_rows:
        text = f"我可能把馆方英文题名输成了“{typo}”。请找出对应记录，并返回规范题名和机构。"
        query_id = add_question(
            "synonym_bilingual_typo",
            text,
            judgment_mode="deterministic_field_gold",
            expected="supported",
            rule={"field": "title", "operator": "single_adjacent_transposition_target", "canonicalValue": obj["title"], "queryValue": typo},
        )
        qrels.append(
            _gold_qrel(query_id, obj, rule="generated adjacent-letter transposition maps to frozen title", matched_fields=["title"])
        )

    for english_type, chinese_type in TYPE_ALIASES[:10]:
        candidates = [
            obj
            for obj in objects
            if _normalise(obj.get("type")).casefold() == english_type
            and _safe_text(obj.get("accessionNumber"), maximum=45)
            and _safe_text(obj.get("institution"), maximum=80)
        ]
        obj = _stable(candidates, f"bilingual-type:{english_type}")[0]
        text = (
            f"我说的是“{chinese_type}”，馆方 type 写作“{english_type}”。"
            f"请在 {obj['institution']} 中定位馆藏号 {obj['accessionNumber']}。"
        )
        query_id = add_question(
            "synonym_bilingual_typo",
            text,
            judgment_mode="deterministic_field_gold",
            expected="supported",
            rule={"fields": ["type", "institution", "accessionNumber"], "operator": "alias_plus_exact_identifiers", "alias": chinese_type, "canonicalType": english_type, "institution": obj["institution"], "accessionNumber": obj["accessionNumber"]},
        )
        qrels.append(
            _gold_qrel(query_id, obj, rule="type alias + institution/accession exact", matched_fields=["type", "institution", "accessionNumber"])
        )

    material_alias_groups: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    for english, chinese in MATERIAL_ALIASES:
        for pack in PACK_LABELS:
            matches = [
                obj
                for obj in objects
                if pack in obj.get("culturePackIds", [])
                and english in _normalise(obj.get("material") or obj.get("medium")).casefold()
            ]
            if 2 <= len(matches) <= 18:
                material_alias_groups.append((english, chinese, pack, matches))
    for english, chinese, pack, matches in _stable(
        material_alias_groups,
        "material-alias",
        key=lambda row: f"{row[0]}:{row[2]}",
    )[:10]:
        text = (
            f"我把材质说成“{chinese}”，英文记录常写“{english}”。"
            f"请找出 culturePackIds 含{PACK_LABELS[pack]}且 material/medium 包含该英文词的对象。"
        )
        query_id = add_question(
            "synonym_bilingual_typo",
            text,
            judgment_mode="deterministic_field_gold",
            expected="supported",
            required_legs=[pack],
            rule={"fields": ["culturePackIds", "material", "medium"], "operator": "pack_membership_and_casefold_contains", "values": [pack, english]},
        )
        qrels.extend(
            _gold_qrel(query_id, obj, rule="culture pack membership + material token contains", matched_fields=["culturePackIds", "material", "medium"])
            for obj in matches
        )

    # 3) Date/material/region filters: 35 exhaustive deterministic groups.
    filter_groups: list[tuple[int, int, str, str, list[dict[str, Any]]]] = []
    for start in range(-1000, 2000, 100):
        end = start + 99
        for english, _ in MATERIAL_ALIASES:
            for pack in PACK_LABELS:
                matches = [
                    obj
                    for obj in objects
                    if pack in obj.get("culturePackIds", [])
                    and english in _normalise(obj.get("material") or obj.get("medium")).casefold()
                    and isinstance(obj.get("dateEarliest"), int)
                    and isinstance(obj.get("dateLatest"), int)
                    and obj["dateEarliest"] <= end
                    and obj["dateLatest"] >= start
                ]
                if 2 <= len(matches) <= 14:
                    filter_groups.append((start, end, english, pack, matches))
    for start, end, material, pack, matches in _stable(
        filter_groups,
        "date-material-region",
        key=lambda row: f"{row[0]}:{row[2]}:{row[3]}",
    )[:35]:
        text = (
            f"请筛出年代范围与 {_date_range_label(start, end)}相交、material/medium 含“{material}”、"
            f"且 culturePackIds 含{PACK_LABELS[pack]}的对象。"
        )
        query_id = add_question(
            "date_material_region",
            text,
            judgment_mode="deterministic_field_gold",
            expected="supported",
            required_legs=[pack],
            rule={"fields": ["dateEarliest", "dateLatest", "material", "medium", "culturePackIds"], "operator": "date_overlap_and_material_contains_and_pack_membership", "values": {"start": start, "end": end, "material": material, "culturePackId": pack}},
        )
        qrels.extend(
            _gold_qrel(query_id, obj, rule="date overlap + material token + culture pack", matched_fields=["dateEarliest", "dateLatest", "material", "medium", "culturePackIds"])
            for obj in matches
        )

    def add_pooled(
        category: str,
        text: str,
        terms: Sequence[str],
        *,
        required_legs: Sequence[str] = (),
    ) -> bool:
        pool = _select_candidates(
            scored_pool(terms),
            required_legs=required_legs,
            limit=12,
        )
        if len(pool) < 5:
            return False
        if required_legs and any(
            not any(leg in row[0].get("culturePackIds", []) for row in pool)
            for leg in required_legs
        ):
            return False
        query_id = add_question(
            category,
            text,
            judgment_mode="pooled_silver_pending_human_review",
            required_legs=required_legs,
            pool_terms=terms,
        )
        qrels.extend(_pooled_qrel(query_id, row) for row in pool)
        return True

    # 4) Open themes: select the first 35 viable frozen specifications.
    for text, terms in OPEN_THEME_SPECS:
        if category_indices["open_theme"] >= CATEGORY_COUNTS["open_theme"]:
            break
        add_pooled("open_theme", text, terms)

    # 5) Cross-cultural questions: choose three actually represented culture legs.
    cross_templates = [
        "比较{packs}馆藏中的{label}：材料和制作记录有哪些可比之处？",
        "{packs}的对象记录怎样分别描述{label}，哪些差异仍需人工解释？",
        "如果把{packs}的{label}并置，馆方字段能支持哪些观察，不能支持哪些联系？",
        "从{packs}各选对象观察{label}，怎样保持每个文化腿都有直接记录？",
        "{label}在{packs}的馆藏元数据中呈现出哪些不同的结构线索？",
    ]
    for label, terms in CROSS_CULTURAL_TOPICS:
        if category_indices["cross_cultural"] >= CATEGORY_COUNTS["cross_cultural"]:
            break
        scored = scored_pool(terms)
        per_pack = {
            pack: sum(1 for obj, _, _ in scored if pack in obj.get("culturePackIds", []))
            for pack in PACK_LABELS
        }
        legs = [pack for pack, count in sorted(per_pack.items(), key=lambda item: (-item[1], item[0])) if count >= 2][:3]
        if len(legs) < 3:
            continue
        packs_text = "、".join(PACK_LABELS[leg] for leg in legs)
        template = cross_templates[category_indices["cross_cultural"] % len(cross_templates)]
        add_pooled(
            "cross_cultural",
            template.format(packs=packs_text, label=label),
            terms,
            required_legs=legs,
        )

    # 6) Symbolic, causal and multi-hop questions: pending human evidence review.
    for text, terms in MULTIHOP_SPECS:
        if category_indices["symbolic_causal_multihop"] >= CATEGORY_COUNTS["symbolic_causal_multihop"]:
            break
        add_pooled("symbolic_causal_multihop", text, terms)

    # 7) Visual motifs: metadata pools only, never visual-relevance gold.
    for text, terms in VISUAL_SPECS:
        if category_indices["visual_motif"] >= CATEGORY_COUNTS["visual_motif"]:
            break
        add_pooled("visual_motif", text, terms)

    # 8) Twelve ambiguous preference probes plus thirteen deterministic evidence-boundary negatives.
    for text, terms in AMBIGUOUS_SPECS:
        add_pooled("ambiguous_out_of_scope", text, terms)
    for text in OUT_OF_SCOPE_QUESTIONS:
        add_question(
            "ambiguous_out_of_scope",
            text,
            judgment_mode="deterministic_evidence_boundary_no_relevant_objects",
            expected="unsupported",
            rule={"operator": "outside_frozen_object_record_evidence_scope", "qrels": "intentionally_empty"},
        )

    actual_counts = Counter(question["category"] for question in questions)
    if dict(actual_counts) != CATEGORY_COUNTS:
        raise RuntimeError(
            f"category generation incomplete: expected {CATEGORY_COUNTS}, got {dict(actual_counts)}"
        )
    if len(questions) != 250:
        raise RuntimeError(f"expected 250 questions, got {len(questions)}")

    manifest = {
        "schemaVersion": "1.0",
        "benchmarkId": "retrieval_eval_v1",
        "version": BENCHMARK_VERSION,
        "status": "frozen",
        "frozenAt": FROZEN_AT,
        "language": "zh-CN with controlled English catalogue terms",
        "questionCount": len(questions),
        "categoryDistribution": CATEGORY_COUNTS,
        "relevanceThreshold": 2,
        "provenance": {
            "collectionId": collection_manifest["id"],
            "collectionVersion": collection_manifest["version"],
            "objectCount": collection_manifest["objectCount"],
            "objectsSha256": collection_manifest["objectsSha256"],
            "collectionManifest": "data/collections/global_open/manifest.json",
            "objectsFile": "data/collections/global_open/objects.json",
            "builder": "scripts/build_retrieval_eval_dataset.py",
            "generationMethod": "deterministic field enumeration plus unjudged metadata/evidence candidate pooling; no LLM used",
        },
        "judgmentPolicy": {
            "deterministic_field_gold": "Exhaustive matches to explicit frozen fields and operators. This is programmatic gold, not human semantic judgement.",
            "pooled_silver_pending_human_review": "Candidate only. relevance is fixed to 0 and evaluator excludes the row from every quality metric until a human creates a separate reviewed judgement.",
            "deterministic_evidence_boundary_no_relevant_objects": "The requested evidence type is outside object records; qrels intentionally empty and answerability is unsupported.",
            "humanGoldPresent": False,
        },
        "pooling": {
            "channels": ["title", "type", "material", "tags", "themes", "description", "evidence"],
            "maximumCandidatesPerQuestion": 12,
            "crossCulturalReservation": "up to two candidates per required culture leg before global fill",
            "warning": "Pooled-silver candidates are discovery aids, not relevance labels.",
        },
        "freezeRules": [
            "Questions and deterministic rules must not be edited after observing run results.",
            "Any human review must be stored as a versioned additive judgement set; pooled-silver rows must not be silently relabelled.",
            "Every qrel object/evidence ID must exist in the frozen global_open hash.",
            "Legacy debugging examples and exact prior QA questions are forbidden.",
            "Changing collection hash, templates, pool terms or category counts requires a new benchmark version.",
        ],
        "excludedLegacyPatterns": LEGACY_BANNED_PATTERNS,
        "files": {},
    }
    return questions, qrels, manifest


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    questions, qrels, manifest = build_dataset()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions_path = OUTPUT_DIR / "questions.jsonl"
    qrels_path = OUTPUT_DIR / "qrels.jsonl"
    _write_jsonl(questions_path, questions)
    _write_jsonl(qrels_path, qrels)
    statuses = Counter(row["judgmentStatus"] for row in qrels)
    modes = Counter(row["judgmentMode"] for row in questions)
    manifest["judgmentDistribution"] = dict(modes)
    manifest["qrelDistribution"] = dict(statuses)
    manifest["files"] = {
        "questions": {
            "path": "data/qa/retrieval_eval_v1/questions.jsonl",
            "sha256": _sha256(questions_path),
            "rows": len(questions),
        },
        "qrels": {
            "path": "data/qa/retrieval_eval_v1/qrels.jsonl",
            "sha256": _sha256(qrels_path),
            "rows": len(qrels),
        },
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"questions       {len(questions)}")
    print(f"qrels           {len(qrels)} {dict(statuses)}")
    print(f"categories      {dict(Counter(row['category'] for row in questions))}")
    print(f"output          {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
