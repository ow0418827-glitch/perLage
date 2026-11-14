"""仅比较洗发水完整图；临时光晕门槛不用于主程序替换。"""
import json
from pathlib import Path
import argparse
import numpy as np
import torch
import compare_final as full
ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');args=p.parse_args()
    scope=full.read(ROOT/'TEMPORARY_SCOPE.json')
    if scope['groups']!=['07_shampoo']:raise ValueError('临时实验只能包含洗发水')
    plan=full.read(ROOT/'PLAN.json')
    plan['cases']=[c for c in plan['cases'] if c['key']=='07_shampoo']
    plan['execution_order']=[['dominant','07_shampoo'],['original','07_shampoo']]
    images=full.freeze_all(plan)
    rows=full.calculate(plan,images)
    result_path=ROOT/'SHAMPOO_TEMPORARY_COMPARISON.json'
    if args.verify:
        previous=full.read(result_path)
        errors=[]
        for a,b in zip(rows,previous['rows']):
            if [a[k] for k in ['group','arm','encoder','image_sha256']]!=[b[k] for k in ['group','arm','encoder','image_sha256']]:raise ValueError('独立复核身份不符')
            errors.extend(abs(a[k]-b[k]) for k in ['image_cosine','joint_mean'])
        if len(rows)!=6 or max(errors)>1e-6:raise ValueError('最终相似度独立复核失败')
        check=dict(passed=True,records=len(rows),scores=len(errors),maximum_absolute_error=max(errors),new_encoder_process=True)
        full.save(ROOT/'SHAMPOO_TEMPORARY_INDEPENDENT_VERIFICATION.json',check)
        report(previous,check)
        return
    differences=[]
    for encoder in [2,1]:
        rr={r['arm']:r for r in rows if r['encoder']==encoder}
        differences.append(dict(encoder=encoder,image_gain=rr['dominant']['image_cosine']-rr['original']['image_cosine'],joint_gain=rr['dominant']['joint_mean']-rr['original']['joint_mean']))
    completion=full.read(ROOT/'runs/dominant/07_shampoo/FULL_METHOD_COMPLETED.json')
    quality=completion['quality'];parent=quality['parent_halo_error']
    result=dict(status='completed_full_shampoo_temporary_relaxed_experiment',scope=scope,rows=rows,
        comparisons=differences,quality=quality,
        strict_halo_limit=max(parent*.5,.5),temporary_halo_limit=max(parent*2.,.5),
        would_pass_original_halo_limit=quality['halo_error']<=max(parent*.5,.5),
        observed_positive_image_gain=all(d['image_gain']>0 for d in differences),
        decision='experiment_only_original_default_method_unchanged',
        main_program_modified=False,other_groups_evaluated=False,private_memory_used=False,shopping_runs=0)
    full.save(result_path,result)


def report(result,check):
    lines=['公子，本轮只完成并比较洗发水组。新方案按用户授权，仅在本次将印刷边缘光晕上限由原初始化值的0.5倍放宽至2倍；原方法对照及主程序仍使用原标准。','',
           '本轮三十步指令生成及四十、四十、二十、六十步初始化优化在首次队列中已完成；本次从这份经过摘要核验的本轮初始化继续，完整执行最终六十步潜变量细化。未导入旧项目或此前试改色图片作训练起点。','',
           '牙刷及其他组未继续生成或评估。相似度只在洗发水新方案与原方法完整图均冻结后计算，并另起进程复核。','',
           '|编码器|未修改原图|原方法完整图|临时放宽主色方案完整图|图像相似度增量|联合分数增量|',
           '|---|---:|---:|---:|---:|---:|']
    for encoder in [2,1]:
        rr={r['arm']:r for r in result['rows'] if r['encoder']==encoder};d=next(x for x in result['comparisons'] if x['encoder']==encoder)
        lines.append(f"|{'主编码器' if encoder==2 else '复核编码器'}|{rr['source']['image_cosine']:.6f}|{rr['original']['image_cosine']:.6f}|{rr['dominant']['image_cosine']:.6f}|{d['image_gain']:+.6f}|{d['joint_gain']:+.6f}|")
    q=result['quality']
    lines += ['',f"新完整图结构相似度：{q['source_ssim']:.6f}；平均绝对误差：{q['source_mae_levels']:.6f}。",'',
              f"实际光晕误差：{q['halo_error']:.6f}；原默认上限：{result['strict_halo_limit']:.6f}；本次临时上限：{result['temporary_halo_limit']:.6f}。",'',
              '若按原默认边缘标准判断：'+('通过。' if result['would_pass_original_halo_limit'] else '未通过；不能将本次放宽条件下的结果写成默认质量标准下的成功。'),'',
              f"独立复核通过：{check['scores']}个分数，最大误差{check['maximum_absolute_error']:.9g}。",'',
              '本轮仅为洗发水、临时放宽边缘检查的完整生成对照；原方法对照保持默认检查，因此这是不同边缘约束下的方案对照，不把差异全部归因于取色本身。没有购物或私有记忆检索。','',
              '主程序默认标准与旧实验均保持原样，不根据本次临时放宽结果自动替换默认方法。最终图片与记录在本轮独立目录保留。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
