"""可训练的风格标签分类候选，全部基于 Python 标准库。

三个候选刻意覆盖不同归纳偏置：

* :class:`MajorityPriorClassifier` 朴素基线：完全不看商品特征，只输出训练集先验，
  用来证明“换季失效”不是换个复杂模型就能掩盖的，任何上线模型必须先跑赢它；
* :class:`LogisticRegressionClassifier` 带 L2 正则与类别权重的多项逻辑回归，
  学习标准化数值特征与面料 one-hot 的线性边界；
* :class:`GaussianNaiveBayesClassifier` 高斯朴素贝叶斯，对小样本和类不平衡更敏感，
  作为第二个真正使用特征的对照模型。

模型只通过 ``predict_proba`` 暴露概率，拒识决策留在评估阶段，不在训练内部写死。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

CANDIDATE_MAJORITY = "majority_prior_baseline"
CANDIDATE_LOGISTIC = "weighted_logistic_regression"
CANDIDATE_NAIVE_BAYES = "gaussian_naive_bayes"
CANDIDATE_IDS = (CANDIDATE_MAJORITY, CANDIDATE_LOGISTIC, CANDIDATE_NAIVE_BAYES)


def class_weights(labels: Sequence[str]) -> dict[str, float]:
    """不平衡类按反比频率加权，避免稀有类被多数类淹没。"""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    total = len(labels)
    return {
        label: total / (len(counts) * count)
        for label, count in counts.items()
    }


def _softmax(scores: Sequence[float]) -> list[float]:
    ceiling = max(scores)
    exps = [math.exp(score - ceiling) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


class MajorityPriorClassifier:
    """朴素基线：预测训练集加权先验，与特征无关。"""

    candidate_id = CANDIDATE_MAJORITY
    uses_features = False

    def __init__(self) -> None:
        self.classes: tuple[str, ...] = ()
        self.prior: dict[str, float] = {}

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[str],
            weights: Sequence[float] | None = None) -> "MajorityPriorClassifier":
        weighted: dict[str, float] = {}
        weights = list(weights) if weights is not None else [1.0] * len(labels)
        for label, weight in zip(labels, weights):
            weighted[label] = weighted.get(label, 0.0) + weight
        total = sum(weighted.values()) or 1.0
        self.classes = tuple(sorted(weighted))
        self.prior = {label: weighted[label] / total for label in self.classes}
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[dict[str, float]]:
        return [dict(self.prior) for _ in rows]

    def to_bundle(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "classes": list(self.classes),
            "prior": self.prior,
        }

    @classmethod
    def from_bundle(cls, bundle: Mapping[str, Any]) -> "MajorityPriorClassifier":
        model = cls()
        model.classes = tuple(bundle["classes"])
        model.prior = dict(bundle["prior"])
        return model


@dataclass(slots=True)
class LogisticRegressionClassifier:
    """带 L2 正则、类别权重的多项逻辑回归（批量梯度下降）。"""

    learning_rate: float = 0.15
    l2_penalty: float = 0.02
    max_iterations: int = 400
    tolerance: float = 1e-6
    candidate_id: str = CANDIDATE_LOGISTIC
    uses_features: bool = True
    classes: tuple[str, ...] = field(default_factory=tuple)
    weights: list[list[float]] = field(default_factory=list)
    bias: list[float] = field(default_factory=list)
    iterations_used: int = 0

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[str],
            weights: Sequence[float] | None = None) -> "LogisticRegressionClassifier":
        self.classes = tuple(sorted(set(labels)))
        if not self.classes:
            raise ValueError("训练标签为空，无法拟合逻辑回归")
        n_features = len(rows[0])
        n_classes = len(self.classes)
        class_index = {label: index for index, label in enumerate(self.classes)}
        sample_weights = list(weights) if weights is not None else [1.0] * len(labels)
        self.weights = [[0.0] * n_features for _ in range(n_classes)]
        self.bias = [0.0] * n_classes

        previous_loss = math.inf
        for iteration in range(1, self.max_iterations + 1):
            gradient_w = [[0.0] * n_features for _ in range(n_classes)]
            gradient_b = [0.0] * n_classes
            loss = 0.0
            for row, label, sample_weight in zip(rows, labels, sample_weights):
                scores = [
                    self.bias[k] + sum(self.weights[k][j] * row[j] for j in range(n_features))
                    for k in range(n_classes)
                ]
                probabilities = _softmax(scores)
                target = class_index[label]
                loss -= sample_weight * math.log(max(probabilities[target], 1e-12))
                for k in range(n_classes):
                    residual = sample_weight * (
                        probabilities[k] - (1.0 if k == target else 0.0)
                    )
                    gradient_b[k] += residual
                    for j in range(n_features):
                        gradient_w[k][j] += residual * row[j]
            scale = 1.0 / max(len(rows), 1)
            for k in range(n_classes):
                self.bias[k] -= self.learning_rate * gradient_b[k] * scale
                for j in range(n_features):
                    regularized = gradient_w[k][j] * scale + self.l2_penalty * self.weights[k][j]
                    self.weights[k][j] -= self.learning_rate * regularized
            loss = loss * scale + 0.5 * self.l2_penalty * sum(
                value * value for matrix in self.weights for value in matrix
            )
            self.iterations_used = iteration
            if abs(previous_loss - loss) < self.tolerance:
                break
            previous_loss = loss
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[dict[str, float]]:
        output: list[dict[str, float]] = []
        for row in rows:
            scores = [
                self.bias[k] + sum(self.weights[k][j] * row[j] for j in range(len(row)))
                for k in range(len(self.classes))
            ]
            probabilities = _softmax(scores)
            output.append(dict(zip(self.classes, probabilities)))
        return output

    def to_bundle(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "classes": list(self.classes),
            "learning_rate": self.learning_rate,
            "l2_penalty": self.l2_penalty,
            "max_iterations": self.max_iterations,
            "iterations_used": self.iterations_used,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_bundle(cls, bundle: Mapping[str, Any]) -> "LogisticRegressionClassifier":
        model = cls(
            learning_rate=float(bundle["learning_rate"]),
            l2_penalty=float(bundle["l2_penalty"]),
            max_iterations=int(bundle["max_iterations"]),
        )
        model.classes = tuple(bundle["classes"])
        model.weights = [list(row) for row in bundle["weights"]]
        model.bias = list(bundle["bias"])
        model.iterations_used = int(bundle.get("iterations_used", 0))
        return model


@dataclass(slots=True)
class GaussianNaiveBayesClassifier:
    """类别加权的高斯朴素贝叶斯，数值特征独立、类别特征随 one-hot 同样按高斯处理。"""

    var_smoothing: float = 1e-9
    candidate_id: str = CANDIDATE_NAIVE_BAYES
    uses_features: bool = True
    classes: tuple[str, ...] = field(default_factory=tuple)
    means: dict[str, list[float]] = field(default_factory=dict)
    variances: dict[str, list[float]] = field(default_factory=dict)
    log_prior: dict[str, float] = field(default_factory=dict)

    def fit(self, rows: Sequence[Sequence[float]], labels: Sequence[str],
            weights: Sequence[float] | None = None) -> "GaussianNaiveBayesClassifier":
        self.classes = tuple(sorted(set(labels)))
        if not self.classes:
            raise ValueError("训练标签为空，无法拟合朴素贝叶斯")
        sample_weights = list(weights) if weights is not None else [1.0] * len(labels)
        n_features = len(rows[0])
        epsilon = self.var_smoothing * max(
            (max(row[j] for row in rows) - min(row[j] for row in rows))
            for j in range(n_features)
        ) if rows else 0.0
        epsilon = max(epsilon, self.var_smoothing)
        total_weight = sum(sample_weights)
        for label in self.classes:
            member = [
                (row, weight)
                for row, item_label, weight in zip(rows, labels, sample_weights)
                if item_label == label
            ]
            weight_sum = sum(weight for _, weight in member) or 1.0
            means = [
                sum(row[j] * weight for row, weight in member) / weight_sum
                for j in range(n_features)
            ]
            variances = [
                sum(weight * (row[j] - means[j]) ** 2 for row, weight in member) / weight_sum
                + epsilon
                for j in range(n_features)
            ]
            self.means[label] = means
            self.variances[label] = variances
            self.log_prior[label] = math.log(weight_sum / total_weight)
        return self

    def predict_proba(self, rows: Sequence[Sequence[float]]) -> list[dict[str, float]]:
        output: list[dict[str, float]] = []
        for row in rows:
            log_scores: dict[str, float] = {}
            for label in self.classes:
                score = self.log_prior[label]
                for j, value in enumerate(row):
                    variance = self.variances[label][j]
                    mean = self.means[label][j]
                    score -= 0.5 * math.log(2.0 * math.pi * variance)
                    score -= ((value - mean) ** 2) / (2.0 * variance)
                log_scores[label] = score
            ceiling = max(log_scores.values())
            exps = {label: math.exp(score - ceiling) for label, score in log_scores.items()}
            total = sum(exps.values()) or 1.0
            output.append({label: exps[label] / total for label in self.classes})
        return output

    def to_bundle(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "classes": list(self.classes),
            "var_smoothing": self.var_smoothing,
            "means": self.means,
            "variances": self.variances,
            "log_prior": self.log_prior,
        }

    @classmethod
    def from_bundle(cls, bundle: Mapping[str, Any]) -> "GaussianNaiveBayesClassifier":
        model = cls(var_smoothing=float(bundle["var_smoothing"]))
        model.classes = tuple(bundle["classes"])
        model.means = {label: list(value) for label, value in bundle["means"].items()}
        model.variances = {label: list(value) for label, value in bundle["variances"].items()}
        model.log_prior = dict(bundle["log_prior"])
        return model


CANDIDATE_BUILDERS = {
    CANDIDATE_MAJORITY: MajorityPriorClassifier,
    CANDIDATE_LOGISTIC: LogisticRegressionClassifier,
    CANDIDATE_NAIVE_BAYES: GaussianNaiveBayesClassifier,
}


def model_from_bundle(bundle: Mapping[str, Any]) -> Any:
    """按冻结清单恢复模型，不读取任何当前标签，保证旧版推断可复现。"""
    builder = CANDIDATE_BUILDERS[str(bundle["candidate_id"])]
    return builder.from_bundle(bundle)
