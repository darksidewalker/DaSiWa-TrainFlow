package trainer

import (
	"os"
	"path/filepath"
)

type Settings struct {
	Architecture              string  `json:"architecture"`
	ProjectName               string  `json:"project_name"`
	TriggerWord               string  `json:"trigger_word"`
	AutoTrigger               bool    `json:"auto_trigger"`
	DatasetPath               string  `json:"dataset_path"`
	OutputPath                string  `json:"output_path"`
	DiTPath                   string  `json:"dit_path"`
	CheckpointPath            string  `json:"checkpoint_path"`
	QwenPath                  string  `json:"qwen_path"`
	VAEPath                   string  `json:"vae_path"`
	NetworkRank               int     `json:"network_rank"`
	LearningRate              string  `json:"learning_rate"`
	UNetLR                    string  `json:"unet_lr"`
	TextEncoderLR1            string  `json:"text_encoder_lr1"`
	TextEncoderLR2            string  `json:"text_encoder_lr2"`
	Optimizer                 string  `json:"optimizer"`
	TrainingSteps             int     `json:"training_steps"`
	SaveSteps                 int     `json:"save_steps"`
	SampleSteps               int     `json:"sample_steps"`
	PositivePrompt            string  `json:"pos_prompt"`
	SamplePrompts             []string `json:"sample_prompts"`
	NegativePrompt            string  `json:"neg_prompt"`
	Width                     int     `json:"width"`
	Height                    int     `json:"height"`
	SampleStepsGen            int     `json:"sample_steps_gen"`
	SampleCFG                 float64 `json:"sample_cfg"`
	SampleSeed                int     `json:"sample_seed"`
	TrainSeed                 int     `json:"train_seed"`
	TrainBatchSize            int     `json:"train_batch_size"`
	GradientAccumulationSteps int     `json:"gradient_accumulation_steps"`
	TargetVRAMPercent         int     `json:"target_vram_percent"`
	TrainUNetOnly             bool    `json:"train_unet_only"`
	FlashAttention            bool    `json:"flash_attention"`
	ResumeEnabled             bool    `json:"resume_enabled"`
	AutoResume                bool    `json:"auto_resume"`
	ResumePath                string  `json:"resume_path"`
	SideMin                   int     `json:"side_min"`
	SideMax                   int     `json:"side_max"`
	TaggerGenThreshold        float64 `json:"tagger_gen_thresh"`
	TaggerCharThreshold       float64 `json:"tagger_char_thresh"`
	TaggerOverwrite           bool    `json:"tagger_overwrite"`
	TargetEpochs              int     `json:"target_epochs"`
	MixedPrecision            string  `json:"mixed_precision"`
	NumCPUThreads             int     `json:"num_cpu_threads"`
	VideoWidth                int     `json:"video_width"`
	VideoHeight               int     `json:"video_height"`
	VideoFPS                  int     `json:"video_fps"`
	VideoDuration             string  `json:"video_duration"`
	VideoTargetFrames         string  `json:"video_target_frames"`
	VideoFrameExtraction      string  `json:"video_frame_extraction"`
	VideoNumRepeats           int     `json:"video_num_repeats"`
	VideoCaptionExtension     string  `json:"video_caption_extension"`
	VideoEnableBucket         bool    `json:"video_enable_bucket"`
	VideoCodec                string  `json:"video_codec"`
	VideoQuality              string  `json:"video_quality"`
	VideoEncoderPreset        string  `json:"video_encoder_preset"`
	VideoSpeed                string  `json:"video_speed"`
	VideoSkipFrames           int     `json:"video_skip_frames"`
	VideoParallelWorkers      int     `json:"video_parallel_workers"`
	VideoIncludeAudio         bool    `json:"video_include_audio"`
	VideoExtraArgs            string  `json:"video_extra_args"`
	LTXVersion                string  `json:"ltx_version"`
	LTXMode                   string  `json:"ltx_mode"`
	LTXVersionCheckMode       string  `json:"ltx_version_check_mode"`
	WanTask                   string  `json:"wan_task"`
	BlocksToSwap              int     `json:"blocks_to_swap"`
	NetworkAlpha              int     `json:"network_alpha"`
	NetworkModule             string  `json:"network_module"`
	TimestepSampling          string  `json:"timestep_sampling"`
	DiscreteFlowShift         string  `json:"discrete_flow_shift"`
	FP8Base                   bool    `json:"fp8_base"`
	FP8Scaled                 bool    `json:"fp8_scaled"`
	SDPA                      bool    `json:"sdpa"`
	GradientCheckpointing     bool    `json:"gradient_checkpointing"`
	UsePinnedMemoryBlockSwap  bool    `json:"use_pinned_memory_for_block_swap"`
	PersistentWorkers         bool    `json:"persistent_data_loader_workers"`
	SaveStateOnTrainEnd       bool    `json:"save_state_on_train_end"`
	MetadataAuthor            string  `json:"metadata_author"`
	MetadataTags              string  `json:"metadata_tags"`
	ExtraTrainArgs            string  `json:"extra_train_args"`
	ExtraCacheTextArgs        string  `json:"extra_cache_text_args"`
	ExtraCacheLatentsArgs     string  `json:"extra_cache_latents_args"`
}

func DefaultSettings(root string) Settings {
	home := defaultSettingsPath(root)
	return Settings{
		Architecture:              ArchitectureAnima,
		ProjectName:               "",
		TriggerWord:               "",
		AutoTrigger:               true,
		DatasetPath:               home,
		OutputPath:                "",
		DiTPath:                   home,
		CheckpointPath:            home,
		QwenPath:                  home,
		VAEPath:                   home,
		NetworkRank:               48,
		LearningRate:              "1e-4",
		UNetLR:                    "1e-4",
		TextEncoderLR1:            "1e-5",
		TextEncoderLR2:            "1e-5",
		Optimizer:                 "Prodigy",
		TrainingSteps:             1100,
		SaveSteps:                 100,
		SampleSteps:               100,
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
		TargetVRAMPercent:         90,
		TrainUNetOnly:             true,
		FlashAttention:            false,
		ResumeEnabled:             false,
		AutoResume:                true,
		ResumePath:                "",
		SideMin:                   512,
		SideMax:                   768,
		TaggerGenThreshold:        0.35,
		TaggerCharThreshold:       0.85,
		TaggerOverwrite:           false,
		TargetEpochs:              6,
		MixedPrecision:            "bf16",
		NumCPUThreads:             8,
		VideoWidth:                768,
		VideoHeight:               512,
		VideoFPS:                  24,
		VideoDuration:             "5",
		VideoTargetFrames:         "1,65,129",
		VideoFrameExtraction:      "full",
		VideoNumRepeats:           1,
		VideoCaptionExtension:     ".txt",
		VideoEnableBucket:         true,
		VideoCodec:                "libx264",
		VideoQuality:              "19",
		VideoEncoderPreset:        "medium",
		VideoSpeed:                "1.0",
		VideoSkipFrames:           0,
		VideoParallelWorkers:      1,
		VideoIncludeAudio:         false,
		LTXVersion:                "2.3",
		LTXMode:                   "video",
		LTXVersionCheckMode:       "error",
		WanTask:                   "i2v-A14B",
		BlocksToSwap:              14,
		NetworkAlpha:              32,
		FP8Base:                   true,
		FP8Scaled:                 true,
		SDPA:                      true,
		GradientCheckpointing:     true,
		UsePinnedMemoryBlockSwap:  true,
		PersistentWorkers:         true,
		SaveStateOnTrainEnd:       true,
		MetadataAuthor:            "darksidewalker",
	}
}

func defaultSettingsPath(root string) string {
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		return filepath.ToSlash(home)
	}
	return filepath.ToSlash(root)
}

type ImageItem struct {
	Src   string `json:"src"`
	Name  string `json:"name"`
	Step  int    `json:"step,omitempty"`
	Label string `json:"label,omitempty"`
}

type StartResponse struct {
	OK           bool   `json:"ok"`
	Message      string `json:"message"`
	PreparedPath string `json:"prepared_path,omitempty"`
	Step         string `json:"step,omitempty"`
}

type AutoCalcResponse struct {
	OK       bool     `json:"ok"`
	Message  string   `json:"message"`
	Settings Settings `json:"settings"`
}
