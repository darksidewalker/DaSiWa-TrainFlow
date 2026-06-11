package trainer

type Settings struct {
	TriggerWord               string  `json:"trigger_word"`
	DatasetPath               string  `json:"dataset_path"`
	DiTPath                   string  `json:"dit_path"`
	QwenPath                  string  `json:"qwen_path"`
	VAEPath                   string  `json:"vae_path"`
	NetworkRank               int     `json:"network_rank"`
	LearningRate              string  `json:"learning_rate"`
	Optimizer                 string  `json:"optimizer"`
	TrainingSteps             int     `json:"training_steps"`
	SaveSteps                 int     `json:"save_steps"`
	SampleSteps               int     `json:"sample_steps"`
	PositivePrompt            string  `json:"pos_prompt"`
	NegativePrompt            string  `json:"neg_prompt"`
	Width                     int     `json:"width"`
	Height                    int     `json:"height"`
	SampleStepsGen            int     `json:"sample_steps_gen"`
	SampleCFG                 float64 `json:"sample_cfg"`
	SampleSeed                int     `json:"sample_seed"`
	TrainSeed                 int     `json:"train_seed"`
	TrainBatchSize            int     `json:"train_batch_size"`
	GradientAccumulationSteps int     `json:"gradient_accumulation_steps"`
	TrainUNetOnly             bool    `json:"train_unet_only"`
	ResumeEnabled             bool    `json:"resume_enabled"`
	AutoResume                bool    `json:"auto_resume"`
	ResumePath                string  `json:"resume_path"`
	SideMin                   int     `json:"side_min"`
	SideMax                   int     `json:"side_max"`
	TaggerGenThreshold        float64 `json:"tagger_gen_thresh"`
	TaggerCharThreshold       float64 `json:"tagger_char_thresh"`
	TaggerOverwrite           bool    `json:"tagger_overwrite"`
}

func DefaultSettings(root string) Settings {
	return Settings{
		TriggerWord:               "",
		DatasetPath:               "",
		DiTPath:                   joinSlash(root, "models", "anima", "dit", "anima-preview.safetensors"),
		QwenPath:                  joinSlash(root, "models", "anima", "text_encoder", "qwen_3_06b_base.safetensors"),
		VAEPath:                   joinSlash(root, "models", "anima", "vae", "qwen_image_vae.safetensors"),
		NetworkRank:               32,
		LearningRate:              "1.0",
		Optimizer:                 "Prodigy",
		TrainingSteps:             2400,
		SaveSteps:                 300,
		SampleSteps:               300,
		PositivePrompt:            "",
		NegativePrompt:            "worst quality, low quality, score_1, score_2, score_3, artist name",
		Width:                     1024,
		Height:                    1024,
		SampleStepsGen:            30,
		SampleCFG:                 4.0,
		SampleSeed:                42,
		TrainSeed:                 42,
		TrainBatchSize:            1,
		GradientAccumulationSteps: 1,
		TrainUNetOnly:             true,
		ResumeEnabled:             false,
		AutoResume:                true,
		ResumePath:                "",
		SideMin:                   512,
		SideMax:                   768,
		TaggerGenThreshold:        0.35,
		TaggerCharThreshold:       0.85,
		TaggerOverwrite:           false,
	}
}

type ImageItem struct {
	Src  string `json:"src"`
	Name string `json:"name"`
}

type StartResponse struct {
	OK      bool   `json:"ok"`
	Message string `json:"message"`
}
