import pytest
from src.experiments.paired_metrics import paired_rank_metrics


def rows(ranks,label='none'):
    return [dict(condition={'label':label},queries=[dict(task_id=str(i),rank=r) for i,r in enumerate(ranks)])]


def test_ceiling_does_not_hide_rank_gain_or_claim_new_recall():
    result=paired_rank_metrics(rows([3,3]),rows([2,2]))
    assert result==dict(n=2,new_top3=0,lost_top3=0,top3_gain=0,mean_rank_gain=1)


def test_new_and_lost_hits_are_matched_by_identity_not_row_order():
    clean=rows([4,2,3]);modified=rows([2,5,2]);modified[0]['queries'].reverse()
    result=paired_rank_metrics(clean,modified)
    assert result['new_top3']==1 and result['lost_top3']==1 and result['top3_gain']==0 and result['mean_rank_gain']==0
    with pytest.raises(ValueError):paired_rank_metrics(clean,rows([2,5]))
    with pytest.raises(ValueError):paired_rank_metrics(clean,rows([2,5,2],label='other'))
