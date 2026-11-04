import unittest
import ast
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .core import ROOT, RankOne, choose_threshold, content_mask, hoyer, split_data, RecordedGate


class CoreTests(unittest.TestCase):
    def test_small_evaluation_budget_covers_all_concept_pairs(self):
        from .core import evaluation_jobs
        concepts = [f'c{i}' for i in range(10)]
        rows = pd.DataFrame([dict(row_id=f'{c}-{i}', concept=c, question=f'{c}-{i}')
                             for c in concepts for i in range(20)])
        jobs = evaluation_jobs(rows, concepts)
        self.assertEqual(len(jobs), 200)
        self.assertEqual(jobs.row_id.nunique(), 200)
        self.assertEqual(jobs.concept.eq(jobs.target).sum(), 100)
        self.assertEqual(len(jobs.groupby(['concept', 'target'])), 100)
        self.assertEqual(jobs.groupby('target').size().tolist(), [20] * 10)

    def test_axbench_prompts_match_upstream_verbatim(self):
        from .axbench_judge import TEMPLATES
        source = ROOT.parent / 'ref/repos/axbench/axbench/evaluators/prompt_templates.py'
        tree = ast.parse(source.read_text())
        values = {node.targets[0].id: ast.literal_eval(node.value)
                  for node in tree.body if isinstance(node, ast.Assign)}
        for axis, original in [('concept_relevance', 'CONCEPT_RELEVANCE'),
                               ('instruction_relevance', 'INSTRUCTION_RELEVANCE'), ('fluency', 'FLUENCY')]:
            self.assertEqual(TEMPLATES[axis], values[f'UNIDIRECTIONAL_PAIRWISE_EVALUATION_{original}_TEMPLATE'])

    def test_axbench_ratings_remain_zero_to_two(self):
        from .axbench_judge import aggregate, parse_rating
        self.assertEqual(parse_rating('Fluent. Rating: [[2]]'), (2., True))
        self.assertEqual(parse_rating('Rating: [[1]]'), (1., True))
        self.assertEqual(parse_rating('Bad answer'), (0., False))
        self.assertEqual(parse_rating('Rating: [[3]]'), (0., False))
        self.assertAlmostEqual(aggregate([2., 1., 2.]), 1.5)
        self.assertEqual(aggregate([2., 0., 2.]), 0.)

    def test_small_training_counts(self):
        from .train import training_examples
        rows = pd.DataFrame([dict(concept=f'c{c}', question=f'{c}-{i}', baseline_output='answer', split='train')
                             for c in range(10) for i in range(10)])
        cfg = dict(train_negative=10, refusal_training_questions=50, split_seed=42)
        self.assertEqual(len(training_examples(rows, 'c0', cfg)), 20)
        refusal = training_examples(rows, '__refusal__', cfg)
        self.assertEqual(len(refusal), 100)
        self.assertEqual([q for q, a in refusal[:50]], [q for q, a in refusal[50:]])

    def test_judge_temperature_matches_axbench_evaluation_entrypoint(self):
        from .core import config
        source = ROOT.parent / 'ref/repos/axbench/axbench/scripts/evaluate.py'
        tree = ast.parse(source.read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and
                 isinstance(n.func, ast.Name) and n.func.id == 'LanguageModel']
        temperatures = [ast.literal_eval(k.value) for n in calls for k in n.keywords if k.arg == 'temperature']
        self.assertEqual(temperatures, [config()['judge_temperature']])

    def test_baseline_summaries_remain_paired_with_intervention_subsets(self):
        from .plot import evaluation_summary
        rows = pd.DataFrame([dict(label=label, concept='cat', target=target, axbench_concept_relevance=score,
                                  axbench_instruction_relevance=2., axbench_fluency=2., axbench_aggregate=score)
                             for label in ['baseline', 'intervention'] for target, score in [('cat', 0.), ('dog', 2.)]])
        summary = evaluation_summary(rows).set_index(['label', 'subset'])
        self.assertEqual(summary.loc[('baseline', 'targeted'), 'axbench_concept_relevance'], 0.)
        self.assertEqual(summary.loc[('baseline', 'untargeted'), 'axbench_concept_relevance'], 2.)

    def test_finished_generations_do_not_accumulate_gate_logs(self):
        op = RecordedGate(torch.tensor([1., 0.]), torch.tensor([0., 1.]), .5, 1.,
                          torch.ones(2, 2, dtype=torch.bool))
        op.alive[0] = False
        out = op(torch.ones(2, 1, 2))
        torch.testing.assert_close(out[0], torch.zeros(1, 2))
        self.assertEqual(op.summary()[0]['decode_gate_fraction'], 0.)
        self.assertEqual(op.summary()[1]['decode_gate_fraction'], 1.)

    def test_common_prefix_exclusion_with_left_padding(self):
        mask = content_mask(torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), 2)
        torch.testing.assert_close(mask, torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.bool))

    def test_inactive_direction_produces_no_intervention(self):
        op = RecordedGate(torch.tensor([1., 0.]), torch.tensor([0., 1.]), .5, 0.,
                          torch.ones(1, 2, dtype=torch.bool))
        torch.testing.assert_close(op(torch.ones(1, 2, 2)), torch.zeros(1, 2, 2))

    def test_unpadded_forward_matches_axbench_source(self):
        source = ROOT.parent / 'ref/repos/axbench/axbench/models/interventions.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TopKReLUIntervention')
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        namespace = {'torch': torch, 'InterventionOutput': SimpleNamespace}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), 'exec'), namespace)
        local = RankOne(4, 2)
        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
        upstream = namespace['forward'](SimpleNamespace(proj=SimpleNamespace(weight=local.weight[None])),
                                        hidden, subspaces={'k': 2})
        result, penalty = local(hidden, torch.ones(2, 3, dtype=torch.bool))
        torch.testing.assert_close(result, upstream.output)
        torch.testing.assert_close(penalty, upstream.latent[1].mean())

    def test_batched_targets_do_not_share_gate_decisions(self):
        detectors = torch.tensor([[1., 0.], [0., 1.]])
        op = RecordedGate(detectors, torch.tensor([0., 1.]), torch.tensor([.5, .5]),
                          torch.tensor([2., 3.]), torch.ones(2, 2, dtype=torch.bool))
        hidden = torch.tensor([[[1., 0.], [0., 0.]], [[1., 0.], [0., 1.]]])
        expected = torch.tensor([[[0., 2.], [0., 0.]], [[0., 0.], [0., 3.]]])
        torch.testing.assert_close(op(hidden), expected)

    def test_topk_matches_explicit_formula_and_has_gradient(self):
        op = RankOne(4, top_k=2)
        with torch.no_grad():
            op.weight.copy_(torch.tensor([0.5, -0.2, 0.1, 0.3]))
        hidden = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
        result, penalty = op(hidden, mask)
        latent = torch.relu(hidden @ op.weight) * mask
        values = latent.topk(2, dim=1).values
        expected = hidden + values.mean(1)[:, None, None] * op.weight * mask[:, :, None]
        torch.testing.assert_close(result, expected)
        (result.square().mean() + penalty).backward()
        self.assertGreater(op.weight.grad.norm().item(), 0)

    def test_padding_does_not_set_training_strength(self):
        op = RankOne(3, top_k=1)
        x = torch.ones(1, 2, 3)
        padded = torch.cat([x, torch.full((1, 1, 3), 1000)], dim=1)
        a, _ = op(x, torch.ones(1, 2, dtype=torch.bool))
        b, _ = op(padded, torch.tensor([[1, 1, 0]], dtype=torch.bool))
        torch.testing.assert_close(a, b[:, :2])

    def test_split_has_no_question_leakage(self):
        train = pd.DataFrame([{'concept': c, 'question': f'{c}-{i}', 'answer': str(i)}
                              for c in ['a', 'b'] for i in range(12)])
        test = train.copy()
        test['question'] += '-test'
        cfg = {'split_seed': 42, 'train_positive': 4,
               'validation_per_concept': 3, 'test_per_concept': 3}
        rows = split_data(train, test, cfg)
        sets = [set(rows[rows['split'] == s].question) for s in ['train', 'validation', 'test']]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(rows.groupby(['split', 'concept']).size().tolist(), [3, 3, 4, 4, 3, 3])

    def test_threshold_uses_strict_gate_comparison(self):
        tau = choose_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        np.testing.assert_array_equal(np.array([0.1, 0.2, 0.8, 0.9]) > tau, [0, 0, 1, 1])

    def test_gate_is_conditional_and_padding_is_excluded_from_logs(self):
        op = RecordedGate(torch.tensor([1., 0.]), torch.tensor([0., 1.]),
                          0.5, 2., torch.tensor([[0, 1, 1]], dtype=torch.bool))
        out = op(torch.tensor([[[9., 0.], [0.1, 0.], [1., 0.]]]))
        torch.testing.assert_close(out[0, 1], torch.zeros(2))
        torch.testing.assert_close(out[0, 2], torch.tensor([0., 2.]))
        self.assertEqual(op.summary()[0]['prefill_gate_fraction'], 0.5)

    def test_hoyer_limits(self):
        self.assertAlmostEqual(hoyer(np.ones(10)), 0)
        self.assertAlmostEqual(hoyer(np.eye(10)[0]), 1)
        self.assertTrue(np.isnan(hoyer(np.zeros(10))))


if __name__ == '__main__':
    unittest.main()
