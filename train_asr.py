from transformers import Trainer
from CustomTrainer import CustomTrainer
from transformers import TrainingArguments


def train_asr(model, data_collator, processor, tokenizer,feature_extractor, train, test, compute_metrics, config):
    
    training_args = TrainingArguments(
            seed=config.seed,
            # output_dir=f"/home/selinawisco/selina_main/asr/models/word_weighting/m3_{MODE}_asr_finetuning_weighted_word_{MODE}_CER",
            output_dir=f"/home/selinawisco/selina_main/asr/models/main/debug_{config.asr_mode}_asr_finetuning_{config.loss_feature}_{config.seed}",
            group_by_length=True,
            # per_device_train_batch_size=1,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            gradient_checkpointing=False, #True,
            # gradient_checkpointing_kwargs={'use_reentrant':True},
            evaluation_strategy="steps",
            num_train_epochs=config.epochs,
            fp16=True,
            save_steps=100,
            eval_steps=1000,
            logging_steps=10,
            learning_rate = 3e-4,
            warmup_steps=500,
            save_total_limit=2,
            # push_to_hub = True,
        )


        

    trainer = CustomTrainer(
        config=config,
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=processor.feature_extractor,
    )


    trainer.train()
    # trainer.train(resume_from_checkpoint='/home/selinawisco/selina_main/asr/models/word_weighting/m3_human_asr_finetuning_weighted_word_human_CER_from_whisper/checkpoint-18100')
    tokenizer.save_pretrained(training_args.output_dir) 
    feature_extractor.save_pretrained(training_args.output_dir)
    model.save_pretrained(training_args.output_dir)
    
    trainer.evaluate()
    

