from transformers import (
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)

from dataset_loader import DentalDataset
from dinov2_semantic_segmentation import Dinov2ForSemanticSegmentation
from segmentation_utils import get_transformation_with_albumentations, SegmentationDataCollator, \
    compute_metrics


def train(
        model_output_path="segout3",
        train_dataset="datasets/dental_train/Images",
        validation_dataset="datasets/dental_val/Images",
        base_model_name="StanfordAIMI/dinov2-base-xray-224",
        batch_size=8,
        gradient_accumulation_steps=4,
        epoches=30,
        mixed_precision=False,
        remove_unused_columns=False,
        save_strategy="epoch",
        learning_rate=5e-5,
        warmup_ratio=0.1,
        logging_steps=20,
        push_to_hub=False,
        internal_segmentation_size=32,
        training_kwargs={},
        trainer_kwargs={},
):
    image_processor = AutoImageProcessor.from_pretrained(base_model_name)
    model = Dinov2ForSemanticSegmentation.from_pretrained(
        base_model_name,
        num_labels=2,  # background + target
        ignore_mismatched_sizes=True,
        task_specific_params={
            "internal_segmentation_size": internal_segmentation_size,
        },
    )

    # transformations = get_transformations(image_processor, aug=False)
    # transformations_with_aug = get_transformations(image_processor, aug=True)
    transformations = get_transformation_with_albumentations(
        image_processor=image_processor, aug=False
    )
    transformations_with_aug = get_transformation_with_albumentations(
        image_processor, aug=True
    )

    train_dataset = DentalDataset(
        path=train_dataset,
        transform=transformations_with_aug,
        size=image_processor.size["height"],
    )
    validation_dataset = DentalDataset(
        path=validation_dataset,
        transform=transformations,
        size=image_processor.size["height"],
    )

    training_args = TrainingArguments(
        output_dir=model_output_path,
        remove_unused_columns=remove_unused_columns,
        save_strategy=save_strategy,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoches,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        push_to_hub=push_to_hub,
        fp16=mixed_precision,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="dice",
        evaluation_strategy="epoch",
        **training_kwargs,
    )
    data_collator = SegmentationDataCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        tokenizer=image_processor,
        **trainer_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    import fire

    fire.Fire(train)
