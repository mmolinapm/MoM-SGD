import json,math
from pathlib import Path
import torch
from experiments import pareto_d10_final as p
from experiments import cifar10_siopt_final as c
ROOT=Path(__file__).resolve().parents[1]
def test_pareto_frozen_alpha_and_budgets():
 assert p.PARETO_ALPHA==2.1 and math.isclose(p.THEORETICAL_SD,math.sqrt(21))
 assert len(p.validate_settings())==420
 for name in p.SCHEDULES:
  for k in p.K_VALUES:assert 0<p.budget_row(name,k)['T_equal_samples']<3000
def test_cnn_rng_and_saved_initialization():
 torch.set_num_threads(2)
 for seed in (42,123,2026):
  config=json.loads((ROOT/f'results/cnn/seed{seed}/config.json').read_text(encoding='utf-8-sig'))['identity']
  with torch.random.fork_rng():
   torch.manual_seed(seed);assert c.state_digest(c.model('cnn').state_dict())==config['initial_sha256']
  state=torch.get_rng_state().clone();schedule=c.schedule(seed)
  assert torch.equal(state,torch.get_rng_state())
  assert c.digest_tensor(schedule)==config['schedule_sha256']
def test_independent_source_tree():
 assert not (ROOT/'experiments/cifar10_fixed_budget_sensitivity.py').exists()
 assert c.e.__file__==str(ROOT/'src/cifar10_utils.py')

def test_original_pareto_data_fingerprints():
    for seed_id,(data_seed,_) in p.SEED_PAIRS.items():
        expected=json.loads((ROOT/f'results/pareto/data_fingerprints_seed{seed_id}.json').read_text())
        raw,noise=p.generate_underlying_data(data_seed)
        assert p.digest(raw)==expected['X_raw_sha256']
        assert p.digest(noise)==expected['epsilon_sha256']
