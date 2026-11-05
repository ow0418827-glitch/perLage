"""以条件与请求标识严格配对的检索增量；不把高基线算成新增命中。"""


def paired_rank_metrics(clean,modified):
    def collect(rows):
        result={}
        for row in rows:
            for q in row['queries']:
                key=(row['condition']['label'],q['task_id'])
                if key in result or type(q['rank']) is not int or q['rank']<1:raise ValueError('配对名次须有效且条件请求不能重复')
                result[key]=q['rank']
        return result
    a,b=collect(clean),collect(modified)
    if not a or set(a)!=set(b):raise ValueError('原图和修改图的条件与请求不能缺失或错配')
    n=len(a);new=sum(a[k]>3 and b[k]<=3 for k in a);lost=sum(a[k]<=3 and b[k]>3 for k in a)
    return dict(n=n,new_top3=new,lost_top3=lost,top3_gain=(new-lost)/n,
        mean_rank_gain=sum(a[k]-b[k] for k in a)/n)
