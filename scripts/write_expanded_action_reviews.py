"""Persist manual image observations for the root reviewer's 13 cases."""
import json
from pathlib import Path

BASE = Path('/home/yujun/program/hermes/results/myobservations/dp_failure_audit_20261010/expanded_review_120')
DATA = [
    (18, 1, 'motion', 'present_in_sampled_input',
     'INPUT 122.122–132.132秒白卡在切割机内前后移动；134.134–136.136秒已换棕色卡，刀头位置继续变化。SOURCE 136.987–137.988秒刀头继续横向移动。',
     '采样中有切割过程和换卡，DP答暂停而H/L答启动新切割。启动与持续切割的精确边界仍需密集运动核对；不能从H正确推断DP删除了刀头证据。'),
    (202, 2, 'motion', 'present_in_sampled_input',
     'INPUT 178–188秒在圆形材料上压刮；190、192秒双手把透明名字贴片向圆片贴合。SOURCE 192.5–193秒仍为贴合动作。',
     '正确选项贴标签得到输入尾部支持；DP选揭起模板，与当前动作方向不符。动作阶段识别或历史干扰均为候选，KV状态未查。'),
    (252, 2, 'annotation', 'present_in_sampled_input',
     'INPUT 252–256秒杯子和木棒位于右上模格并倒入金色混合物；258秒木棒已经移到右侧中间格，SOURCE 258.5–259秒也在中间格。',
     'gold描述刚才上格倒金色树脂，DP描述截点附近中格用棒调整；两者均在合法输入出现。right now标注窗口与动作快速切换存在疑点，尚不能认定gold错误。'),
    (347, 2, 'sampling_gap', 'local_sampling_gap',
     'INPUT 204–206秒把面包条放在烤盘；208秒手伸向左边油瓶但尚未见淋油。SOURCE 209–210秒近景才清楚显示向面包淋油。',
     '问题问摆面包后的动作，局部关键淋油片段落在最后采样与截点之间。H答对可能来自推断，不能据此证明H缓存了未采样的尾帧；更早历史是否出现过相同行为未全查。'),
    (361, 4, 'motion', 'present_in_sampled_input',
     'INPUT 390.390秒拿起Brad名字；392.392–394.394秒将带名字贴片放下并按压。396.396秒后转向Emily，402.402秒已处理其他名字。',
     '带Brad实体线索的正确历史动作已采到，DP取揭保护层这一相邻阶段。需要绑定名字与过去动作；没有最终KV跟踪，保留损失与读取干扰仍未分离。'),
    (384, 4, 'motion', 'present_in_sampled_input',
     'INPUT 412、414秒在挑除多余贴纸；416秒翻起转印层，418秒举起清理后的透明名字贴片。SOURCE 418.5–419秒继续揭起贴片。',
     '正确的清理后揭起动作已在418秒输入出现，DP停留在更早挑除多余材料。动作阶段更新或历史干扰候选，不能归为已证实采样漏失。'),
    (84, 0, 'motion', 'present_in_sampled_input',
     'INPUT 14、16秒刀尖沿棕色圆片表面工作；18秒棕色覆盖层已被掀起，SOURCE 18.5–19秒继续剥离。',
     'DP答沿线切割，对应前一阶段；剥离的状态变化已进入输入。更密集输入或动作状态表示可能有帮助，但本轮未做重放干预。'),
    (465, 1, 'motion', 'present_in_sampled_input',
     'INPUT 62秒蓝积木已放下；64秒绿积木置于蓝块旁；66秒手拿橙块，68秒橙块已放到其旁。',
     '绿块后的拿橙块事件清楚进入采样。三方法均答调整绿块，支持顺序绑定/阶段更新候选；最终缓存中事件证据是否完整保留未知。'),
    (388, 2, 'detail_binding', 'present_in_sampled_input',
     'INPUT 244秒双手持透明材料，246–250秒在紫色圆片旁压刮；254–258秒双手固定圆片上的透明带图案材料，SOURCE 259–260秒相同。',
     '透明图案贴片在输入可见；DP描述白色材料贴紫色圆片，与背景/早期底纸外观相近。细粒度材料和被固定对象的绑定候选，不能证明视觉token删除。'),
    (378, 0, 'sampling_gap', 'local_sampling_gap',
     '最后INPUT 24秒仍展示桌上的空透明球、漏斗和闪粉管，没有操作中的手；SOURCE 25–26秒切到双手旋转球体，使粉色闪粉覆盖内壁。',
     '当前旋转动作未进入末采样。已看尾部与锚点，未穷尽全部早期帧，不称全前缀无任何相关物体证据。'),
    (350, 1, 'sampling_gap', 'local_sampling_gap',
     'INPUT 54–58秒仍是Cricut Maker 3切割透明材料；SOURCE 59–60秒才切到灰台面摆放粉色切片和串珠，60秒字幕Let us assemble。',
     '三方法的切割机答案符合末输入，gold对应下一镜头。此局部截点缺口发生于压缩前，需合法截点补帧重放才能量化修复。'),
    (377, 3, 'detail_binding', 'present_in_sampled_input',
     'INPUT 484–498秒连续显示画笔在头盔右侧刷粉色背景，SOURCE 499–500秒仍在刷粉色。更早0、98秒确为白底上蓝/棕色绘画阶段。',
     '当前粉色背景和早期蓝棕阶段都采到，DP选择早期描述。时间范围just now与颜色/区域绑定失败候选，原始证据存在不代表KV已保留。'),
    (422, 4, 'motion', 'present_in_sampled_input',
     'INPUT 86–96秒人在镜子前戴假发、整理衣领；98秒已切到另一房间红沙发前，男子拿起金色衣物。SOURCE 98.48–99秒继续展开衣物。',
     '场景切换至少已有一帧进入输入，但DP选择前段坐着整理假发。新事件权重、时间绑定或缓存保留均待分离；不是确定的完全未采样。'),
]

def main():
    reviews = []
    for vid, qi, group, status, obs, limit in DATA:
        directory = BASE / 'frames_action_clips' / f'v{vid}_q{qi}'
        paths = [str(directory / name) for name in ('anchors_1.jpg', 'tail_1.jpg')]
        assert all(Path(p).is_file() for p in paths)
        reviews.append(dict(video_id=vid, question_index=qi,
                            primary_group_code=group, input_evidence_status=status,
                            review_scope='tail_and_anchors', observations=obs,
                            interpretation_and_limit=limit, viewed_sheet_paths=paths,
                            kv_retention_verified=False, causal_intervention_run=False))
    assert len(reviews) == len({(r['video_id'], r['question_index']) for r in reviews}) == 13
    out = BASE / 'reviews_action_clips.json'
    out.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {len(reviews)} manually reviewed cases to {out}')

if __name__ == '__main__':
    main()
