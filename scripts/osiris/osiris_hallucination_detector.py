#! /usr/bin/env python


"""
Osiris 7B Hallucination Detector

A production-ready hallucination detection system using the Osiris 7B model
for evaluating LLM outputs on the FRAMES dataset.

Features:
- Supports FRAMES dataset format (prompt, llm_response, ground_truth)
- Handles both contexted and context-free scenarios
- Reproducible results (fixed seed)
- Comprehensive evaluation metrics
- Professional visualizations

Usage:
    from osiris_detector import OsirisDetector

    detector = OsirisDetector()
    results = detector.evaluate("input.json", "output.json")

Requirements:
    - Python 3.8+
    - PyTorch 2.0+
    - transformers>=4.36.0
    - bitsandbytes>=0.41.0
    - See requirements.txt for full list
"""

# Standard Library:
import argparse
import json
from pathlib import Path
from typing import Union, List, Dict, Optional

# 3rd party libraries:
from dataclasses import dataclass, asdict, field
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Visualizations disabled.")

# For notebook inline display
try:
    from IPython.core.getipython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('matplotlib', 'inline')
except:
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MODEL = "judgmentlabs/Qwen2.5-Osiris-7B-Instruct"
SEED = 42
# Note:  Updated context windows (input tokens) to 32k to match model
# MAX_CHARS = 8000  # Reduced to prevent token limit issues (4096 tokens)
MAX_CHARS = 10240  # Use 2.5 * token count, not used as dataset records <= 22k

__version__ = '2025-11-16-01'

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DetectionResult:
    """
    Single detection result

    Attributes:
        sample_id: Unique identifier for the sample
        question: Input question/prompt
        answer: Model's answer to evaluate
        context: Reference context (empty if not provided)
        is_hallucination: Detection verdict (True/False)
        confidence: Confidence score (0.0-1.0)
        osiris_response: Raw model response
        has_context: Whether context was provided
        ground_truth: Optional ground truth label
        metadata: Additional metadata from input
    """
    sample_id: str
    question: str
    answer: str
    context: str
    is_hallucination: bool
    confidence: float
    osiris_response: str
    has_context: bool
    ground_truth: Optional[bool] = None
    metadata: Dict = field(default_factory=dict)

@dataclass
class Metrics:
    """
    Evaluation metrics

    Includes basic detection statistics and performance metrics
    when ground truth labels are available.
    """
    # Basic statistics
    total_samples: int
    hallucinations_detected: int
    hallucination_rate: float   # Proportion of samples detected as hallucinations

    # Data quality
    samples_with_context: int
    samples_without_context: int

    # Classification metrics (when ground truth available)
    accuracy: Optional[float] = None # Number of correct predictions / total predictions
    accuracy2: Optional[float] = None
    accuracy3: Optional[float] = None
    precision: Optional[float] = None # Number of true positives / (true positives + false positives)
    precision2: Optional[float] = None
    precision3: Optional[float] = None
    recall: Optional[float] = None
    recall2: Optional[float] = None
    recall3: Optional[float] = None
    f1_score: Optional[float] = None
    f1_score2: Optional[float] = None
    f1_score3: Optional[float] = None
    true_positives: Optional[int] = None
    false_positives: Optional[int] = None
    true_negatives: Optional[int] = None
    false_negatives: Optional[int] = None

    # Confidence breakdown
    high_confidence_samples: int = 0
    low_confidence_samples: int = 0

# ============================================================================
# VALIDATOR
# ============================================================================

class InputValidator:
    """
    Validates and normalizes input data for the detector

    Handles multiple field name conventions:
    - FRAMES format: prompt, llm_response, ground_truth
    - Generic format: question, answer, label
    """

    @staticmethod
    def validate(data: Union[List[Dict], Dict]) -> tuple:
        """
        Validate and clean input data

        Args:
            data: Input data (dict or list of dicts)

        Returns:
            Tuple of (valid_samples, error_messages)
        """
        if isinstance(data, dict):
            data = [data]

        valid_samples = []
        errors = []

        for i, sample in enumerate(data):
            # Extract with multiple field name support
            question = (
                sample.get('prompt') or sample.get('Prompt') or
                sample.get('question') or sample.get('Question')
            )
            # answer = sample.get("llm_response") or sample.get("answer")
            answer = sample.get('ModelAnswer') or sample.get('llm_response')
            ground_truth = sample.get('Answer') or sample.get('ground_truth')
            if ground_truth is None:
                ground_truth = sample.get("label")
            context = sample.get('Context', '') or sample.get('context', '')

            # Validation
            if not question or not answer:
                errors.append(f"Sample {i}: Missing question/answer")
                continue

            # Truncate long sequences
            '''
            if len(question) > MAX_CHARS:
                print(f'Warning:  Question {i + 1} too long, truncating...')
                question = question[:MAX_CHARS]
            if len(answer) > MAX_CHARS:
                print(f'Warning:  Answer {i + 1} too long, truncating...')
                answer = answer[:MAX_CHARS]
            if context and len(context) > MAX_CHARS:
                print(f'Warning:  Context {i + 1} too long, truncating...')
                context = context[:MAX_CHARS]
            '''

            valid_samples.append({
                "sample_id": sample.get("sample_id", str(i + 1)),
                "question": question.strip(),
                "answer": answer.strip(),
                "context": context.strip() if context else "",
                "ground_truth": bool(ground_truth) if ground_truth is not None else None,
                "metadata": {
                    "evaluation_decision": (
                        sample.get("evaluation_decision") or sample.get('GraderDecision')
                    ),
                    "evaluation_explanation": (
                        sample.get("evaluation_explanation") or sample.get('GraderExplanation')
                    ),
                    "reasoning_type": (
                        sample.get("reasoning_type") or sample.get('ReasoningTypes')
                    )
                }
            })

        return valid_samples, errors

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _safe_div(a: float, b: float) -> float:
    """Safe division that returns 0 if denominator is 0"""
    return a / b if b > 0 else 0.0 # Avoid ZeroDivisionError

# ============================================================================
# OSIRIS DETECTOR
# ============================================================================

class OsirisDetector:
    """
    Osiris 7B Hallucination Detector

    Main interface for detecting hallucinations in LLM outputs using the
    Osiris 7B model with 4-bit quantization.

    Example:
        >>> detector = OsirisDetector()
        >>> results = detector.evaluate("input.json", "output.json")
        >>> print(f"Detected {results['metrics']['hallucinations_detected']} hallucinations")

    Args:
        model_name: Hugging Face model identifier
        seed: Random seed for reproducibility
        verbose: Print progress messages
        use_gpu: Use GPU if available
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, seed: int = SEED,
                 verbose: bool = True, use_gpu: bool = True):
        self.model_name = model_name
        self.seed = seed
        self.verbose = verbose
        self.use_gpu = use_gpu and torch.cuda.is_available()

        self._set_seed()
        self._load_model()

    def _set_seed(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _load_model(self):
        """Load Osiris model with 4-bit quantization"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            if self.verbose:
                print(f"Loading: {self.model_name}")
                print(f"Device: {'GPU' if self.use_gpu else 'CPU'}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Tokenizer defaults to 4k context size:
            print(f'Default tokenizer max length: {self.tokenizer.model_max_length}')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                ),
                device_map="auto" if self.use_gpu else "cpu",
                torch_dtype=torch.float16
            )

            # Tokenizer defaults to 4k context size but model supports 32k, fix:
            self.tokenizer.model_max_length = self.model.config.max_position_embeddings
            print(f'Adjusted tokenizer max length: {self.tokenizer.model_max_length}')

            if self.verbose:
                print("Model loaded successfully\n")

        except ImportError as e:
            raise ImportError(
                "Missing dependencies. Install with:\n"
                "pip install transformers bitsandbytes accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def detect_single(self, question: str, answer: str, context: str = "",
                     sample_id: str = "?") -> DetectionResult:
        """
        Detect hallucination for a single sample

        Args:
            question: Input question/prompt
            answer: Model's answer to evaluate
            context: Reference context (optional)
            sample_id: Sample identifier

        Returns:
            DetectionResult object with verdict and metadata
        """
        # Handle empty answer
        if not answer or not answer.strip():
            return DetectionResult(
                sample_id=sample_id,
                question=question,
                answer=answer,
                context=context,
                is_hallucination=True,
                confidence=0.0,
                osiris_response="empty_answer",
                has_context=bool(context)
            )

        # Build prompt
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Task: Is this answer supported by the context? "
            f"Reply ONLY: 'SUPPORTED' or 'HALLUCINATED'."
        )

        # Generate
        try:
            messages = [
                {"role": "system", "content": "You are a fact-checker."},
                {"role": "user", "content": prompt}
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                from transformers import GenerationConfig
                # Based on values included in model's generation_config.json
                # and recommendations from Hugging Face:
                gen_cfg = GenerationConfig(
                    bos_token_id=151643,
                    do_sample=False,
                    eos_token_id=[151645, 151643],
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.05,
                    max_new_tokens=20,
                )
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_cfg
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            ).strip().lower()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                response = "cuda_oom"
                if self.verbose:
                    print(f"CUDA OOM: {sample_id}")
            else:
                response = "error"
                if self.verbose:
                    print(f"Error in {sample_id}: {e}")
        except Exception as e:
            response = "error"
            if self.verbose:
                print(f"Error in {sample_id}: {e}")

        # Parse response
        is_hall = any(x in response for x in ["hallucinated", "not supported"])
        has_ctx = bool(context and context.strip())

        # Confidence scoring
        if has_ctx:
            confidence = 0.9
        else:
            # Without context, higher confidence if detected hallucination
            confidence = 0.5 if is_hall else 0.3

        return DetectionResult(
            sample_id=sample_id,
            question=question,
            answer=answer,
            context=context,
            is_hallucination=is_hall,
            confidence=confidence,
            osiris_response=response,
            has_context=has_ctx
        )

    def evaluate(self, input_path: str, output_path: str,
                 visualize: bool = True, inline: bool = False) -> Dict:
        """
        Evaluate hallucinations in a dataset

        Args:
            input_path: Path to input JSON file
            output_path: Path to save results JSON
            visualize: Generate visualization plots
            inline: Show plots inline (for notebooks)

        Returns:
            Dictionary containing results and metrics
        """
        if self.verbose:
            print(f"Loading: {input_path}\n")

        # Load data
        try:
            with open(input_path) as f:
                raw_data = json.load(f)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load {input_path}: {e}") from e

        # Data summary
        self._print_data_summary(raw_data)

        # Validate
        samples, errors = InputValidator.validate(raw_data)

        if errors and self.verbose:
            print(f"{len(errors)} validation errors:")
            for err in errors[:5]:
                print(f"  - {err}")
            if len(errors) > 5:
                print(f"  - ...and {len(errors) - 5} more\n")

        if not samples:
            raise ValueError("No valid samples found")

        if self.verbose:
            print(f"Valid samples: {len(samples)}\n")

        # Detect with progress tracking
        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc="Detecting", unit="sample")
        except ImportError:
            iterator = samples
            if self.verbose:
                print("Detecting hallucinations...")

        results = []
        for sample in iterator:
            result = self.detect_single(
                question=sample["question"],
                answer=sample["answer"],
                context=sample["context"],
                sample_id=sample["sample_id"]
            )
            result.ground_truth = sample.get("ground_truth")
            result.metadata = sample.get("metadata", {})
            results.append(result)

        if self.verbose:
            print("\nDetection complete\n")

        # Calculate metrics
        metrics = self._calculate_metrics(results)
        self._print_metrics(metrics)

        # Prepare output
        output = {
            "metadata": {
                "model": self.model_name,
                "seed": self.seed,
                "input_file": input_path,
                "output_file": output_path,
                "total_samples": len(samples),
                "validation_errors": len(errors),
                "version": __version__
            },
            "results": [asdict(r) for r in results],
            "metrics": asdict(metrics)
        }

        # Save
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        if self.verbose:
            print(f"Saved: {output_path}\n")

        # Visualize
        if visualize and VISUALIZATION_AVAILABLE:
            viz_paths = self._visualize(results, metrics, output_path, inline)
            output['visualizations'] = viz_paths
        elif visualize and not VISUALIZATION_AVAILABLE:
            print("Warning: Visualization libraries not available")

        return output

    def _print_data_summary(self, raw_data: List[Dict]):
        """Print dataset summary"""
        print("="*80)
        print("DATASET SUMMARY")
        print("="*80)

        data = raw_data if isinstance(raw_data, list) else [raw_data]
        total = len(data)

        print(f"\nTotal Samples: {total}")

        # Field analysis
        all_fields = set()
        for sample in data:
            if isinstance(sample, dict):
                all_fields.update(sample.keys())

        print(f"\nFields Present: {sorted(all_fields)}")

        # Field completeness
        key_fields = {
            'prompt': 'Question/Prompt',
            'llm_response': 'Model Answer',
            'ground_truth': 'Ground Truth Label',
            'context': 'Reference Context'
        }

        print("\nField Completeness:")
        for field, description in key_fields.items():
            if field in all_fields:
                count = sum(1 for s in data if isinstance(s, dict) and s.get(field))
                pct = _safe_div(count, total) * 100
                print(f"  {description:<25} {count:>4}/{total} ({pct:>5.1f}%)")

        # Length statistics
        for field, name in [('prompt', 'Question'), ('llm_response', 'Answer')]:
            if field in all_fields:
                lengths = [len(str(s.get(field, ''))) for s in data if isinstance(s, dict)]
                if lengths and max(lengths) > 0:
                    print(f"\n{name} Length Stats:")
                    print(f"  Min:    {min(lengths):>6} chars")
                    print(f"  Max:    {max(lengths):>6} chars")
                    print(f"  Mean:   {np.mean(lengths):>6.1f} chars")
                    print(f"  Median: {np.median(lengths):>6.1f} chars")

        # Ground truth distribution
        if 'ground_truth' in all_fields:
            gt_vals = [s.get('ground_truth') for s in data if isinstance(s, dict)]
            gt_vals = [v for v in gt_vals if v is not None]
            if gt_vals:
                true_count = sum(1 for v in gt_vals if v)
                false_count = len(gt_vals) - true_count
                print(f"\nGround Truth Distribution:")
                print(f"  Hallucinations:     {true_count:>4}/{len(gt_vals)} ({_safe_div(true_count, len(gt_vals))*100:>5.1f}%)")
                print(f"  Not Hallucinations: {false_count:>4}/{len(gt_vals)} ({_safe_div(false_count, len(gt_vals))*100:>5.1f}%)")

        # Context availability
        if 'context' in all_fields:
            has_ctx = sum(1 for s in data if isinstance(s, dict) and s.get('context'))
            print(f"\nContext Availability:")
            print(f"  With Context:    {has_ctx:>4}/{total} ({_safe_div(has_ctx, total)*100:>5.1f}%)")
            print(f"  Without Context: {total-has_ctx:>4}/{total} ({_safe_div(total-has_ctx, total)*100:>5.1f}%)")

            if has_ctx == 0:
                print("\n  NOTE: Dataset has no context")
                print("  Osiris performance will be limited without reference context")

        print("="*80 + "\n")

    def _calculate_metrics(self, results: List[DetectionResult]) -> Metrics:
        """Calculate evaluation metrics"""
        total = len(results)
        halls = sum(r.is_hallucination for r in results)
        with_ctx = sum(r.has_context for r in results)
        high_conf = sum(r.confidence > 0.5 for r in results)

        metrics = Metrics(
            total_samples=total,
            hallucinations_detected=halls,
            hallucination_rate=_safe_div(halls, total),
            samples_with_context=with_ctx,
            samples_without_context=total - with_ctx,
            high_confidence_samples=high_conf,
            low_confidence_samples=total - high_conf
        )

        # Ground truth metrics
        ###
        gt_results = [r for r in results if r.ground_truth is not None]
        if gt_results:
            # y_true = [r.ground_truth for r in gt_results]
            # Not does ground_truth exist, but its value (is it a hallucination
            # or 'FALSE'); in other words the grader says the answer is wrong:
            y_true = [r.metadata['evaluation_decision'] == 'TRUE' for r in gt_results]
            y_true2 = [r.metadata['evaluation_decision'] == 'FALSE' for r in gt_results]
            y_pred = [r.is_hallucination for r in gt_results]

            print(f'\nDebugging:\ngt_results#  {len(gt_results)}\ny_true: {y_true}\n'
                  f'y_true2: {y_true2}\ny_pred: {y_pred}\n')
            '''
                                          | Incorrect/Hallucination | Correct/Not a hallucination
            Predicted Hallucination       |   True Positive (TP)    |   False Positive (FP)
            Predicted Not a hallucination |   False Negative (FN)   |   True Negative (TN)

            Ground Truth (GT) = Grader assessment of model's answer (True/False)
            Hallucination Prediction (HP) = Osiris model's prediction (True/False)
            '''
            # True Positive:  GT=False (Wrong answer), HP=True (Hallucination)
            tp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)

            # False Positive:  GT=True (Right answer), HP=True (Hallucination)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t and p)

            # True Negative:  GT=True (Right answer), HP=False (Not a hallucination)
            tn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)

            # False Negative:  GT=False (Wrong answer), HP=False (Not a hallucination)
            fn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

            metrics.true_positives = tp
            metrics.false_positives = fp
            metrics.true_negatives = tn
            metrics.false_negatives = fn

            metrics.accuracy = _safe_div(tp + tn, len(gt_results))
            metrics.precision = _safe_div(tp, tp + fp)
            metrics.recall = _safe_div(tp, tp + fn)
            metrics.f1_score = _safe_div(
                2 * metrics.precision * metrics.recall,
                metrics.precision + metrics.recall
            )
            #
            # Using Scikit-Learn:
            metrics.accuracy2 = accuracy_score(y_true, y_pred)
            metrics.accuracy3 = accuracy_score(y_true2, y_pred)
            metrics.precision2 = precision_score(y_true, y_pred)
            metrics.precision3 = precision_score(y_true2, y_pred)
            metrics.recall2 = recall_score(y_true, y_pred)
            metrics.recall3 = recall_score(y_true2, y_pred)
            metrics.f1_score2 = f1_score(y_true, y_pred)
            metrics.f1_score3 = f1_score(y_true2, y_pred)

            print(f'\nDebugging Using Scikit-Learn:')
            print(f"  Accuracy:  {metrics.accuracy2:.3f}")
            print(f"  Precision: {metrics.precision2:.3f}")
            print(f"  Recall:    {metrics.recall2:.3f}")
            print(f"  F1-Score:  {metrics.f1_score2:.3f}")

            print(f'\nDebugging Using Scikit-Learn Alt:')
            print(f"  Accuracy:  {metrics.accuracy3:.3f}")
            print(f"  Precision: {metrics.precision3:.3f}")
            print(f"  Recall:    {metrics.recall3:.3f}")
            print(f"  F1-Score:  {metrics.f1_score3:.3f}")


            ### Specificity - TN / (TN + FP)
            ### Precision-recall/ROC curve/ROC-AUC, PR-AUC???

        return metrics

    def _print_metrics(self, m: Metrics):
        """Print evaluation metrics"""
        print("="*80)
        print("EVALUATION METRICS")
        print("="*80)
        print(f"\nTotal Samples:              {m.total_samples}")
        print(f"Hallucinations Detected:    {m.hallucinations_detected} ({m.hallucination_rate:.1%})")
        print(f"Samples with Context:       {m.samples_with_context} ({_safe_div(m.samples_with_context, m.total_samples):.1%})")
        print(f"High Confidence Detections: {m.high_confidence_samples} ({_safe_div(m.high_confidence_samples, m.total_samples):.1%})")

        if m.accuracy is not None:
            print(f"\nDetection Performance (vs Ground Truth):")
            print(f"  Accuracy:  {m.accuracy:.3f}")
            print(f"  Precision: {m.precision:.3f}")
            print(f"  Recall:    {m.recall:.3f}")
            print(f"  F1-Score:  {m.f1_score:.3f}")
            print(f"\nConfusion Matrix:")
            print(f"  TP={m.true_positives:<4} FP={m.false_positives:<4}")
            print(f"  FN={m.false_negatives:<4} TN={m.true_negatives:<4}")

        print("="*80 + "\n")

    def _visualize(self, results: List[DetectionResult], metrics: Metrics,
                   output_path: str, inline: bool = False) -> List[str]:
        """Generate visualizations"""
        viz_paths = []
        output_dir = Path(output_path).parent

        sns.set_style("whitegrid")

        # 1. Hallucination Distribution Pie Chart
        fig, ax = plt.subplots(figsize=(8, 6))

        labels = ['Supported', 'Hallucinated']
        sizes = [metrics.total_samples - metrics.hallucinations_detected,
                metrics.hallucinations_detected]
        colors = ['#2ecc71', '#e74c3c']
        explode = (0, 0.1)

        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, explode=explode, shadow=True,
               textprops={'fontsize': 12, 'weight': 'bold'})
        ax.set_title('Hallucination Detection Results',
                    fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        path1 = output_dir / "hallucination_distribution.png"
        plt.savefig(path1, dpi=150, bbox_inches='tight')
        if inline:
            plt.show()
        plt.close()
        viz_paths.append(str(path1))

        # 2. Confusion Matrix (if ground truth available)
        if metrics.accuracy is not None:
            fig, ax = plt.subplots(figsize=(8, 6))
            cm = np.array([[metrics.true_negatives, metrics.false_positives],
                          [metrics.false_negatives, metrics.true_positives]])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Pred: Not Hall', 'Pred: Hall'],
                       yticklabels=['Actual: Not Hall', 'Actual: Hall'],
                       cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
            ax.set_title(f'Confusion Matrix\nAcc: {metrics.accuracy:.1%}, F1: {metrics.f1_score:.3f}',
                        fontsize=14, fontweight='bold')

            plt.tight_layout()
            path2 = output_dir / "confusion_matrix.png"
            plt.savefig(path2, dpi=150, bbox_inches='tight')
            if inline:
                plt.show()
            plt.close()
            viz_paths.append(str(path2))

            # 3. Performance Metrics Bar Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metric_values = [metrics.accuracy, metrics.precision,
                           metrics.recall, metrics.f1_score]
            colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']

            bars = ax.bar(metric_names, metric_values, color=colors,
                         alpha=0.8, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title('Detection Performance Metrics',
                        fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')

            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom',
                       fontweight='bold', fontsize=11)

            plt.tight_layout()
            path3 = output_dir / "performance_metrics.png"
            plt.savefig(path3, dpi=150, bbox_inches='tight')
            if inline:
                plt.show()
            plt.close()
            viz_paths.append(str(path3))

        if self.verbose:
            print(f"Generated {len(viz_paths)} visualizations\n")

        return viz_paths


def run_detector(input_path: str, output_path: str, *, visualize: bool=True,
                 inline: bool=False, verbose: bool=True) -> None:
    detector = OsirisDetector(verbose=verbose)
    results = detector.evaluate(input_path, output_path, visualize=visualize, inline=inline)

    print("="*80)
    print("DETECTION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    print(f"Visualizations: {len(results.get('visualizations', []))} files")
    print(f"Hallucinations detected: {results['metrics']['hallucinations_detected']}")
    print()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for running the detector."""
    parser = argparse.ArgumentParser(
        description="Run the Osiris hallucination detector on a JSON dataset"
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to input JSON file containing samples"
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        help="Path to save the detector results JSON"
    )
    parser.add_argument(
        "--inline",
        action="store_true",
        help="Display matplotlib plots inline (useful in notebooks)"
    )
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Disable visualization generation"
    )
    parser.set_defaults(visualize=True)
    return parser.parse_args()


def main():
    """
    Example usage of the Osiris detector

    This function demonstrates basic usage. Provide
    your own data input/output paths as command-line arguments to actually run.
    """
    args = parse_args()

    if args.input_path and args.output_path:
        input_path = args.input_path
        output_path = args.output_path
    else:
        print("Usage: python osiris_hallucination_detector.py <input.json> <output.json>")
        print("\nNo input/output paths provided. Creating example dataset for demonstration...")

        example = [
            {
                "prompt": "What is the capital of France?",
                "llm_response": "Paris",
                "ground_truth": False,
                "context": "France is a country in Europe. Paris is its capital."
            },
            {
                "prompt": "When was Python created?",
                "llm_response": "Python was created in 1850.",
                "ground_truth": True,
                "context": "Python was created by Guido van Rossum in 1991."
            },
            {
                "prompt": "What is machine learning?",
                "llm_response": "Machine learning is a subset of AI.",
                "ground_truth": False,
                "context": ""
            }
        ]

        input_path = "data/example_input.json"
        output_path = "data/example_output.json"

        with open(input_path, 'w') as f:
            json.dump(example, f, indent=2)

        print(f"Created: {input_path}\n")

    run_detector(input_path, output_path, visualize=args.visualize, inline=args.inline)


if __name__ == "__main__":
    main()
