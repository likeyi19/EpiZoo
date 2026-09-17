"""Small structural fixtures, not substitute weights for runtime acceptance."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from epizoo.models.epizoo import EpiZoo,EpiZooConfig
from epizoo.models.epizoo_x import EpiZooXConfig
from epizoo.qualification import qualify_source,initialize_target,strict_target,parameter_scopes,check_forward
from epizoo.data.ccre import _parse_overlap_map,_write_indexed_bed
from epizoo.data.datasets import CellDatasetX,truncate_cell


@pytest.fixture
def source():
    cfg=EpiZooConfig(vocab_size=11,human_vocab_size=3,mouse_vocab_size=4,
        emb_dim=16,max_rank=32,num_layers=1,num_heads=2,use_flash_attn=False,cca_hidden_dim=8)
    state={k:v.half() for k,v in EpiZoo(cfg).state_dict().items() if not k.endswith('pos_weight')}
    return qualify_source(state,cfg),cfg


def target_cfg():
    return EpiZooXConfig(vocab_size=9,emb_dim=16,max_rank=32,num_layers=1,
        num_heads=2,use_flash_attn=False,cca_hidden_dim=8)


@pytest.mark.parametrize('strategy,species',[('de_novo',None),('mapped_reference','human'),('mapped_reference','mouse')])
def test_transfer(source,strategy,species):
    state,_=source; seq=torch.arange(80,dtype=torch.float32).reshape(5,16)/80
    mapping={0:1,2:1} if species else None
    torch.manual_seed(2)
    model=initialize_target(state,seq,target_cfg(),strategy=strategy,mapping=mapping,source_species=species)
    assert parameter_scopes(model)['frozen']==['seq_emb.weight']
    assert torch.equal(model.seq_emb.weight[4:],seq)
    assert torch.equal(model.rank_emb.weight,state['rank_emb.weight'].float())
    if species:
        offset=4+(3 if species=='mouse' else 0)
        assert torch.equal(model.ccre_emb.weight[4],state['ccre_emb.weight'][1+offset].float())
        assert torch.equal(model.signal_decoder.weight[0],model.signal_decoder.weight[2])
    assert not torch.equal(model.ccre_emb.weight[8],state['ccre_emb.weight'][4].float())
    loaded=strict_target(model.state_dict(),target_cfg())
    assert check_forward(loaded,torch.tensor([[1,4,5,2]]),device='cpu',use_amp=False)['decoder_size']==5


@pytest.mark.parametrize('bad',['missing','unexpected','shape','nan','dtype','buffer'])
def test_source_failures(source,bad):
    state,cfg=source;state=dict(state)
    if bad=='missing':del state['rank_emb.weight']
    elif bad=='unexpected':state['unknown']=torch.ones(1)
    elif bad=='shape':state['rank_emb.weight']=torch.ones(1).half()
    elif bad=='nan':state['rank_emb.weight']=state['rank_emb.weight'].clone();state['rank_emb.weight'][0,0]=float('nan')
    elif bad=='dtype':state['rank_emb.weight']=state['rank_emb.weight'].float()
    else:state['signal_loss_fn.pos_weight']=torch.tensor(1.)
    with pytest.raises(ValueError):qualify_source(state,cfg)


def test_no_fallback(source):
    state,_=source
    for mapping,species in ((None,'human'),({},None),({0:99},'human')):
        with pytest.raises((ValueError,IndexError)):
            initialize_target(state,torch.ones(5,16),target_cfg(),strategy='mapped_reference',mapping=mapping,source_species=species)


def test_overlap_ties_and_original_indices(tmp_path):
    bed=tmp_path/'a.bed';bed.write_text('chr1\t40\t60\nchr1\t0\t20\n')
    indexed=tmp_path/'indexed.bed';_write_indexed_bed(str(bed),indexed,'idx')
    assert indexed.read_text().splitlines()==['chr1\t0\t20\t1','chr1\t40\t60\t0']
    overlap=tmp_path/'overlap.bed'
    overlap.write_text('chr1\t0\t20\t1\tchr1\t0\t20\t2\t20\n'
                       'chr1\t0\t20\t1\tchr1\t0\t20\t0\t20\n'
                       'chr1\t40\t60\t0\tchr1\t45\t55\t1\t10\n')
    assert _parse_overlap_map(overlap,1)=={0:1,1:2}
    assert _parse_overlap_map(overlap,21)=={}


def test_dataset_sampling_targets():
    np.random.seed(19)
    dataset=CellDatasetX([[4,5,6,7]],num_ccres=8,max_length=4,cca_alpha=1,random_sample=True)
    first=dataset[0]
    np.random.seed(19);second=dataset[0]
    assert torch.equal(first['input_ids'],second['input_ids'])
    assert first['signal'].sum()==4
    assert first['cca_labels'].tolist()==[1,1,0,0]
    assert first['input_ids'][0]==1 and first['input_ids'][-1]==2
    assert first['input_ids'][1:-1].tolist()==sorted(first['input_ids'][1:-1].tolist())


def test_empty_overlap(tmp_path):
    p=tmp_path/'empty';p.touch()
    assert _parse_overlap_map(p)=={}


def test_cancellable_child_is_reaped(tmp_path):
    import sys
    import os
    from epizoo.data.ccre import _cancellable_run
    pidfile=tmp_path/'pid'
    def cancel():
        if pidfile.exists():raise InterruptedError('cancelled')
    with pytest.raises(InterruptedError):
        _cancellable_run([sys.executable,'-c',
            'import os,time,sys;open(sys.argv[1],"w").write(str(os.getpid()));time.sleep(30)',str(pidfile)],cancel)
    with pytest.raises(ProcessLookupError):os.kill(int(pidfile.read_text()),0)


def test_public_trainer_objectives_and_checkpoint(source,tmp_path):
    from epizoo.train.posttrain import EpiZooXPostTrainer,EpiZooXPostTrainConfig
    from epizoo.data.datasets import collate_fn_x
    from torch.utils.data import DataLoader
    state,_=source
    model=initialize_target(state,torch.ones(5,16),target_cfg(),strategy='de_novo')
    dataset=CellDatasetX([[4,5],[6,7]],num_ccres=5,max_length=32)
    cfg=EpiZooXPostTrainConfig(output_dir=str(tmp_path),max_steps=2,save_steps=2,log_steps=2,
        device='cpu',use_amp=False,warmup_steps=1)
    trainer=EpiZooXPostTrainer(model,DataLoader(dataset,batch_size=1,collate_fn=collate_fn_x),cfg)
    before=model.seq_emb.weight.detach().clone()
    trainer.train()
    assert trainer.global_step==2 and torch.equal(model.seq_emb.weight,before)
    assert isinstance(trainer.optimizer,torch.optim.AdamW)
    checkpoint=next(tmp_path.glob('*.pth'))
    state=torch.load(checkpoint,weights_only=True)
    assert not {'optimizer','scheduler','rng_state'} & state.keys()
    strict_target(state,target_cfg())


def test_real_mapping_tools_on_nonbiological_fixture(tmp_path):
    from pathlib import Path
    from epizoo.data.ccre import build_ccre_map
    liftover=Path('/home/likeyi/program/liftOver');bedtools=Path('/usr/bin/bedtools')
    if not liftover.is_file() or not bedtools.is_file():pytest.skip('Optional local mapping binaries unavailable')
    new=tmp_path/'new.bed';new.write_text('chrT\t40\t60\nchrT\t5\t25\nchrT\t100\t120\nchrT\t42\t58\n')
    ref=tmp_path/'ref.bed';ref.write_text('chrR\t10\t30\nchrR\t0\t20\nchrR\t40\t60\n')
    chain=tmp_path/'targetToSource.chain';chain.write_text('chain 100 chrT 200 + 0 80 chrR 200 + 0 80 1\n80\n\n')
    result=build_ccre_map(str(new),str(ref),str(chain),str(liftover),str(bedtools),tmp_dir=str(tmp_path))
    assert result=={0:2,1:1,3:2}


@pytest.mark.parametrize('bad',['key','decoder','nan','dtype','buffer'])
def test_target_checkpoint_fail_closed(source,bad):
    state,_=source
    model=initialize_target(state,torch.ones(5,16),target_cfg(),strategy='de_novo')
    target=dict(model.state_dict())
    if bad=='key':target.pop('rank_emb.weight')
    elif bad=='decoder':target['signal_decoder.weight']=torch.ones(4,16)
    elif bad=='nan':target['signal_decoder.bias']=torch.full((5,),float('nan'))
    elif bad=='dtype':target['rank_emb.weight']=target['rank_emb.weight'].half()
    else:target['signal_loss_fn.pos_weight']=torch.tensor(1.)
    with pytest.raises(ValueError):strict_target(target,target_cfg())
