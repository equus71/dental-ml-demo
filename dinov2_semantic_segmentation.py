import torch
from transformers import Dinov2Model, Dinov2PreTrainedModel
from transformers.modeling_outputs import SemanticSegmenterOutput


class LinearClassifier(torch.nn.Module):
    """
    Linear classifier for semantic segmentation using 1x1 convolutions.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    tokenW : int, optional
        Width of token grid, by default 32
    tokenH : int, optional
        Height of token grid, by default 32
    num_labels : int, optional
        Number of output classes, by default 1

    Notes
    -----
    The classifier applies a 1x1 convolution to transform the input features
    into class logits for each spatial position.
    """

    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        """
        Forward pass of the classifier.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embeddings of shape (batch_size, height * width, channels)

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_labels, height, width)
        """
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
        embeddings = embeddings.permute(0, 3, 1, 2)

        return self.classifier(embeddings)


class MLPClassifier(torch.nn.Module):
    """
    MLP classifier for semantic segmentation.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    tokenW : int, optional
        Width of token grid, by default 32
    tokenH : int, optional
        Height of token grid, by default 32
    num_labels : int, optional
        Number of output classes, by default 1

    Notes
    -----
    The classifier applies a two-layer MLP with LayerNorm and GELU activation
    to transform the input features into class logits for each spatial position.
    """

    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(MLPClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels * 2),
            torch.nn.LayerNorm(in_channels * 2),
            torch.nn.GELU(),
            torch.nn.Linear(in_channels * 2, num_labels),
        )

    def forward(self, embeddings):
        """
        Forward pass of the classifier.

        Parameters
        ----------
        embeddings : torch.Tensor
            Input embeddings of shape (batch_size, height * width, channels)

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_labels, height, width)
        """
        # Keep batch dimension and combine height,width dimensions
        embeddings = embeddings.reshape(-1, self.height * self.width, self.in_channels)
        # Apply MLP to each token position
        logits = self.classifier(embeddings)
        # Reshape back to spatial dimensions
        logits = logits.reshape(-1, self.height, self.width, logits.size(-1))
        # Match Conv2d output format (B, C, H, W)
        logits = logits.permute(0, 3, 1, 2)

        return logits


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    """
    DINOv2 model adapted for semantic segmentation tasks.

    This model adds a semantic segmentation head on top of the DINOv2 backbone.
    It uses the patch embeddings from the backbone and upsamples the predictions
    to match the input image resolution.

    Parameters
    ----------
    config : transformers.PretrainedConfig
        Model configuration

    Notes
    -----
    The model uses an MLP classifier by default, which processes each patch
    embedding independently and then upsamples the results to the original
    image resolution using bilinear interpolation.
    """

    def __init__(self, config):
        super().__init__(config)
        internal_segmentation_size = config.task_specific_params.get(
            "internal_segmentation_size", 32
        )

        self.dinov2 = Dinov2Model(config)
        self.classifier = MLPClassifier(
            config.hidden_size,
            internal_segmentation_size,
            internal_segmentation_size,
            config.num_labels,
        )

    def forward(
            self,
            pixel_values,
            output_hidden_states=False,
            output_attentions=False,
            labels=None,
    ):
        """
        Forward pass of the model.

        Parameters
        ----------
        pixel_values : torch.Tensor
            Input images
        output_hidden_states : bool, optional
            Whether to return all hidden states, by default False
        output_attentions : bool, optional
            Whether to return attention weights, by default False
        labels : torch.Tensor, optional
            Ground truth segmentation masks, by default None

        Returns
        -------
        SemanticSegmenterOutput
            Object containing:
            - loss : Optional cross entropy loss if labels provided
            - logits : Segmentation logits
            - hidden_states : Optional hidden states if requested
            - attentions : Optional attention weights if requested
        """
        # use frozen features
        outputs = self.dinov2(
            pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)
        logits = torch.nn.functional.interpolate(
            logits, size=pixel_values.shape[2:], mode="bilinear", align_corners=False
        )

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
