from math import ceil

import numpy as np
import torch
from ada_verona import apply_pytorch_normalization
from scipy.stats import binomtest, norm
from statsmodels.stats.proportion import proportion_confint


# Adapted from https://github.com/locuslab/smoothing/blob/master/code/core.py
class Smooth:
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1
    # misclassified prediction (when sample_correct_predictions=True)
    MISCLASSIFIED = -2

    def __init__(
        self,
        base_classifier: torch.nn.Module,
        num_classes: int,
        sigma: float,
        t: int,
        sample_correct_predictions: bool = True,
    ):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        :param t: diffusion timestep
        :param sample_correct_predictions: if True, only certify correctly classified samples
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.t = t
        self.sample_correct_predictions = sample_correct_predictions

    def certify(
        self,
        x: torch.tensor,
        n0: int,
        n: int,
        alpha: float,
        batch_size: int,
        label: int | None = None,
    ) -> (int, float):
        """Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        be robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :param label: if provided, only certify if prediction matches true label
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # check if prediction matches true label
        if label is not None and cAHat != label and self.sample_correct_predictions:
            return Smooth.MISCLASSIFIED, 0.0
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]

        if binomtest(count1, count1 + count2, p=0.5).pvalue > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def base_predict(self, x: torch.tensor) -> int:
        """Direct prediction on clean input without diffusion denoising or MC sampling.
        This method computes the clean accuracy by calling the base classifier directly
        on the input without any smoothing or denoising steps.
        :param x: the input [channel x height x width]
        :return: the predicted class
        """
        self.base_classifier.eval()
        with torch.no_grad():
            # Add batch dimension: [channel x height x width] -> [1 x channel x height x width]
            x_batch = x.unsqueeze(0)
            
            # Check if base_classifier is a DiffusionRobustModel (has classifier attribute)
            # If so, call the underlying classifier directly without diffusion
            if hasattr(self.base_classifier, "classifier"):
                # This is a DiffusionRobustModel - call classifier directly
                classifier = self.base_classifier.classifier
                classifier_type = getattr(self.base_classifier, "classifier_type", None)
                
                # Preprocess input similar to DiffusionRobustModel.forward but skip denoising
                # Input x is in [0,1] range, keep it as is for preprocessing
                
                # Determine target size based on classifier type
                if classifier_type == "onnx" or classifier_type == "pytorch":
                    target_size = (classifier.expected_height, classifier.expected_width)
                elif classifier_type == "timm":
                    cfg = getattr(classifier, "default_cfg", {}) or {}
                    input_size = cfg.get("input_size", (3, 512, 512))
                    target_size = (input_size[1], input_size[2])
                else:  # huggingface or default
                    target_size = (224, 224)
                
                # Resize to target size (input is in [0,1] range)
                imgs = torch.nn.functional.interpolate(
                    x_batch, target_size, mode="bicubic", antialias=True
                )
    
                if classifier_type == "huggingface":
                    #https://huggingface.co/aaraki/vit-base-patch16-224-in21k-finetuned-cifar10/blob/main/preprocessor_config.json
                    imgs = imgs * 2 - 1
                elif classifier_type == "timm":
                    #https://huggingface.co/timm/beit_large_patch16_512.in22k_ft_in22k_in1k
                    imgs = imgs * 2 - 1
                elif classifier_type == "pytorch":
                    pytorch_normalization = getattr(
                        self.base_classifier, "pytorch_normalization", "none"
                    )
                    imgs = apply_pytorch_normalization(imgs, pytorch_normalization)
                
                out = classifier(imgs)
                logits = out.logits if hasattr(out, "logits") else out
                prediction = logits.argmax(1).item()
            else:
                # Regular classifier - call directly (may need t parameter)
                # Try calling without t first, fall back to with t if needed
                try:
                    out = self.base_classifier(x_batch)
                    if hasattr(out, "logits"):
                        prediction = out.logits.argmax(1).item()
                    else:
                        prediction = out.argmax(1).item()
                except TypeError:
                    # Classifier requires t parameter, use t=0 for clean prediction
                    out = self.base_classifier(x_batch, 0)
                    if hasattr(out, "logits"):
                        prediction = out.logits.argmax(1).item()
                    else:
                        prediction = out.argmax(1).item()
            
            return prediction

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))

                predictions = self.base_classifier(batch, self.t).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            
            # if torch.cuda.is_available():
            #     if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > 0.9:
            #         torch.cuda.empty_cache()
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

