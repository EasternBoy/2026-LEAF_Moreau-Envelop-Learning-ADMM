"""Regression tests for temporal splits, initialization, correction and cache identity."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from DC3.common.runner import build, _reference, check_checkpoint_config
from DC3.common.dc3 import DC3Config, DC3Solver, train_dc3
from DC3.common.completion import LinearCompletion
from DC3.entr_max.problem import MaxEntropyProblem
from DC3.entr_max.data import make_split, StreamingConeParams
from DC3.entr_max.experiment import SPEC as CONE
from DC3.power_grid.experiment import SPEC as POWER, reference_solve
from DC3.power_grid.data import split_offsets, make_split as grid_split
from DC3.power_grid.problem import EcoMPCProblem, GridParams

class ProtocolTests(unittest.TestCase):
    def test_temporal_raw_samples_are_disjoint(self):
        pools = split_offsets(96, 1000, 0, gap=12)
        used = {k: set().union(*(set(range(i,i+96)) for i in v)) for k,v in pools.items()}
        self.assertTrue(used['train'].isdisjoint(used['valid']))
        self.assertTrue(used['train'].isdisjoint(used['test']))
        self.assertTrue(used['valid'].isdisjoint(used['test']))
        self.assertGreater(min(used['test']), max(used['valid'])+12)
        with self.assertRaisesRegex(ValueError, '288'):
            split_offsets(96,192,0)

    def test_temporal_files_offsets_and_no_nominal_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'series.csv'
            path.write_text('timestamp,power\n'+''.join(f'{i},{i}\n' for i in range(400)))
            p=grid_split(96,3,0,'test','cpu',torch.float64,load_csv=str(path),gen_csv=str(path))
            self.assertTrue(bool((p.offsets > 0).all()))
            torch.testing.assert_close(p.load[:,0],p.offsets.double())
            torch.testing.assert_close(p.index(torch.tensor([1])).offsets,p.offsets[1:2])
            other=Path(tmp)/'misaligned.csv';other.write_text(path.read_text().replace('0,0','different,0'))
            with self.assertRaisesRegex(ValueError,'timestamps differ'):
                grid_split(96,1,0,'test','cpu',torch.float64,load_csv=str(path),gen_csv=str(other))

    def test_streaming_pool_reproducible_and_index_independent(self):
        pool=StreamingConeParams(20,3,8000,4,'train','cpu',torch.float64)
        both=pool.index(torch.tensor([27,2]))
        one=pool.index(torch.tensor([2]))
        torch.testing.assert_close(both.A[1:],one.A)
        self.assertEqual(len(pool),8000)
        self.assertFalse(torch.equal(both.A[0],both.A[1]))

    def test_large_initialization_is_uniform_and_preconditioner_is_correct(self):
        cfg=json.loads(Path('DC3/entr_max/configs/default.json').read_text())
        p,c,s,dc,dev,dt=build(CONE,cfg)
        params=make_split(1000,100,3,0,'test',dev,dt)
        for training in [False,True]:
            s.train(training)
            y=s(params)
            torch.testing.assert_close(y,torch.full_like(y,.001),atol=1e-12,rtol=1e-10)
            self.assertTrue(bool(p.domain_valid(params,y).all()))
        self.assertLess(s.net.n_params(),7_000_000)
        d=torch.randn(2,c.n_partial,dtype=dt)
        # For simplex completion, (I + 11')^-1 d = d - sum(d)/n.
        torch.testing.assert_close(s._precondition(d),d-d.sum(1,keepdim=True)/1000)
        z=s.predict_partial(params)
        y=s.correction_output(params,s.correct_train(params,z))
        self.assertTrue(bool(torch.isfinite(y).all()))
        self.assertLess(float(p.eq_resid(params,y).abs().max()),1e-10)

    def test_full_correction_matches_direct_full_gradient_with_and_without_completion(self):
        p=MaxEntropyProblem(4,1)
        params=make_split(4,1,2,0,'test','cpu',torch.float64)
        comp=LinearCompletion(p.A_eq,strategy='explicit',other_vars=[3])
        for completion in [True,False]:
            cfg=DC3Config(use_compl=completion,corr_mode='full',corr_train_steps=1,
                          corr_test_max_steps=1,corr_lr=.001,corr_momentum=0,hidden_size=8,batch_norm=False)
            s=DC3Solver(p,comp,cfg).double()
            z=torch.full((2,3 if completion else 4),.6,dtype=torch.float64,requires_grad=True)
            y=s.complete(params,z)
            yg=y.detach().requires_grad_(True)
            loss=.5*s.ineq_dist_int(params,yg).square().sum()+.5*p.eq_resid(params,yg).square().sum()
            expected=yg-.001*torch.autograd.grad(loss,yg)[0]
            corrected=s.correct_train(params,z)
            torch.testing.assert_close(corrected,expected)
            grad=torch.autograd.grad(corrected.square().sum(),z)[0]
            self.assertTrue(bool(torch.isfinite(grad).all()))
            test,_,_,_=s.correct_test(params,z.detach())
            torch.testing.assert_close(test,expected)
            cfg.use_train_corr=False;cfg.use_test_corr=False
            torch.testing.assert_close(s.correct_train(params,z),y)
            torch.testing.assert_close(s.correct_test(params,z.detach())[0],y)

    def test_reference_cache_identity_and_legacy_invalidation(self):
        params=make_split(4,1,2,0,'test','cpu',torch.float64)
        calls=[]
        def reference(p,cfg):
            calls.append(1)
            return np.ones((len(p),4))/4,np.ones(len(p)),np.ones(len(p)),['optimal']*len(p)
        spec=SimpleNamespace(name='test',reference_solve=reference,build_problem=lambda *a:SimpleNamespace(n_y=4))
        cfg={'problem':{'n':4},'reference':{'tol':1e-9}}
        with tempfile.TemporaryDirectory() as tmp:
            _reference(spec,cfg,params,tmp,False);_reference(spec,cfg,params,tmp,False)
            self.assertEqual(len(calls),1)
            params.b[0,0]+=.1
            _reference(spec,cfg,params,tmp,False)
            cfg['reference']['tol']=1e-8
            _reference(spec,cfg,params,tmp,False)
            cfg['problem']['constant']=2
            _reference(spec,cfg,params,tmp,False)
            self.assertEqual(len(calls),4)
            np.savez(Path(tmp)/'reference.npz',Y=np.zeros((2,4)))
            _reference(spec,cfg,params,tmp,False)
            self.assertEqual(len(calls),5)

    def test_power_reference_uses_overridden_constants(self):
        cfg={'problem':{'N':2,'A':.95,'BESS':750,'r_op':7,'a':30},'reference':{'tol':1e-10}}
        p=POWER.build_problem(cfg['problem'],'cpu',torch.float64)
        params=GridParams(torch.tensor([.5],dtype=torch.float64),torch.full((1,2),40.,dtype=torch.float64),torch.zeros(1,2,dtype=torch.float64))
        y,_,obj,status=reference_solve(params,cfg)
        y=torch.as_tensor(y)
        self.assertIn('optimal',status[0])
        np.testing.assert_allclose(p.obj_fn(params,y,safe=False).numpy(),obj,rtol=1e-8)
        self.assertLess(float(p.eq_resid(params,y).abs().max()),1e-6)
        self.assertLess(float(p.ineq_dist(params,y).max()),1e-6)

    def test_checkpoint_cannot_relabel_legacy_split(self):
        cfg=json.loads(Path('DC3/power_grid/configs/legacy.json').read_text())
        ck={'config':copy.deepcopy(cfg)}
        check_checkpoint_config(ck,cfg,POWER)
        cfg['data']['split_strategy']='temporal'
        with self.assertRaisesRegex(ValueError,'protocol differs'):
            check_checkpoint_config(ck,cfg,POWER)
        cfg=copy.deepcopy(ck['config']);cfg['problem']['BESS']=750
        with self.assertRaisesRegex(ValueError,'problem differs'):
            check_checkpoint_config(ck,cfg,POWER)

if __name__=='__main__': unittest.main()
