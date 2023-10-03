"""
Interactive SD Pipeline using IPython widgets
Author: Andrei Cozma
"""

import time
from pprint import pprint

import diffusers
import ipywidgets as widgets
import torch
from IPython.display import clear_output, display

from . import helpers

wid_width = "95%"
wid_description_width = "15%"
wid_description_width_wide = "30%"

defaults = {
    "model": None,
    # "scheduler": "Whatever's default ¯\_(ツ)_/¯",
    "scheduler": None,
    "prompt": "flock of sheep are having selfie with a grazing on grassland, himalayan background extra detailed, highly realistic, extra detailed, himalayn landscape, hyper realistic",
    "negative_prompt": "low-res, low quality, jpeg artifacts, blurry, grainy, distorted, ugly, out of frame, disfigured, bad anatomy, watermarked",
    "guidance_scale": 7.5,
    "num_inference_steps": 30,
    "num_images_per_prompt": 6,
    "seed": -1,
}


class PipelineControlPanel:
    def __init__(self, **kwargs):
        self.w_model = widgets.Dropdown(
            value=kwargs.get("model") or defaults["model"],
            options=[model.value for model in helpers.ModelsTxt2Img],
            description="model",
            style={"description_width": wid_description_width},
            layout=widgets.Layout(width=wid_width),
        )
        self.w_model.observe(
            lambda value: self.on_model_change(value["new"]), names="value"
        )

        self.w_scheduler = widgets.Dropdown(
            value=kwargs.get("scheduler") or defaults["scheduler"],
            options=[defaults["scheduler"]],
            description="scheduler",
            style={"description_width": wid_description_width},
            layout=widgets.Layout(width=wid_width),
        )

        self.w_prompt = widgets.Textarea(
            value=kwargs.get("prompt") or defaults["prompt"],
            rows=3,
            description="prompt",
            style={"description_width": wid_description_width},
            layout=widgets.Layout(width=wid_width),
        )
        self.w_negative_prompt = widgets.Textarea(
            value=kwargs.get("negative_prompt") or defaults["negative_prompt"],
            rows=3,
            description="negative_prompt",
            style={"description_width": wid_description_width},
            layout=widgets.Layout(width=wid_width),
        )
        self.w_guidance_scale = widgets.FloatSlider(
            value=kwargs.get("guidance_scale") or defaults["guidance_scale"],
            min=0,
            max=30,
            step=0.25,
            description="guidance_scale",
            style={"description_width": wid_description_width_wide},
            layout=widgets.Layout(width=wid_width),
        )

        self.w_num_inference_steps = widgets.IntSlider(
            value=kwargs.get("num_inference_steps") or defaults["num_inference_steps"],
            min=1,
            max=75,
            step=1,
            description="num_inference_steps",
            style={"description_width": wid_description_width_wide},
            layout=widgets.Layout(width=wid_width),
        )
        self.w_num_images_per_prompt = widgets.IntSlider(
            value=kwargs.get("num_images_per_prompt")
            or defaults["num_images_per_prompt"],
            min=1,
            max=8,
            step=1,
            description="num_images_per_prompt",
            style={"description_width": wid_description_width_wide},
            layout=widgets.Layout(width=wid_width),
        )

        self.w_seed = widgets.IntSlider(
            value=kwargs.get("seed") or defaults["seed"],
            min=-1,
            max=100000,
            step=1,
            description="seed",
            style={"description_width": wid_description_width_wide},
            layout=widgets.Layout(width=wid_width),
        )

        self.wb_reset = widgets.Button(
            description="Reset",
            button_style="warning",
            tooltip="Reset the controls to their default values",
            icon="history",
            layout=widgets.Layout(width=wid_width),
        )
        self.wb_reset.on_click(lambda _: self.reset_defaults())

        self.widgets = [
            self.w_model,
            self.w_scheduler,
            self.w_prompt,
            self.w_negative_prompt,
            self.w_guidance_scale,
            self.w_num_inference_steps,
            self.w_num_images_per_prompt,
            self.w_seed,
            self.wb_reset,
        ]

    def on_model_change(self, new_model):
        pass

    @property
    def model(self) -> str:
        return self.w_model.value

    @property
    def scheduler_cls(self):
        return self.w_scheduler.value

    @property
    def prompt(self) -> str:
        return self.w_prompt.value

    @property
    def negative_prompt(self) -> str:
        return self.w_negative_prompt.value

    @property
    def guidance_scale(self) -> float:
        return self.w_guidance_scale.value

    @property
    def num_inference_steps(self) -> int:
        return self.w_num_inference_steps.value

    @property
    def num_images_per_prompt(self) -> int:
        return self.w_num_images_per_prompt.value

    @property
    def seed(self) -> int:
        return self.w_seed.value

    def reset_defaults(self):
        clear_output()
        for control in self.widgets:
            control.value = defaults[control.description[:-1]]

    def show(self, clear=True):
        if clear:
            clear_output()
        self.controls = widgets.GridBox(
            children=self.widgets,
            layout=widgets.Layout(grid_template_columns="repeat(2, 50%)"),
        )
        display(self.controls)


class InteractivePipeline(PipelineControlPanel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__pipe_signature = None
        self.pipe = None
        self.w_model.value = kwargs.get("model") or defaults["model"]

        self.wb_run = widgets.Button(
            description="Run",
            button_style="success",
            tooltip="Run the pipeline with the current settings",
            icon="play",
            layout=widgets.Layout(width=wid_width),
        )

        self.on_run = lambda: self.__call__(interactive=True)
        self.wb_run.on_click(lambda _: self.on_run())
        self.widgets.append(self.wb_run)

    def on_model_change(self, new_model):
        if new_model is None:
            return []

        self.init_pipeline(new_model)

        clear_output()
        self.show()

    def init_pipeline(
        self,
        model=None,
        scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        pipe_signature_comps = (model, scheduler, device)
        pipe_signature = hash(pipe_signature_comps)
        # Re-utilize the current pipeline if the signature is the same
        if pipe_signature == self.__pipe_signature:
            return

        assert model is not None, "Model must be specified."

        # Delete the current pipeline
        del self.pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load the pretrained model
        self.pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            # low_cpu_mem_usage=True,
        )
        print(f"Model: {model}")

        # Set the scheduler
        if scheduler:
            self.pipe.scheduler = scheduler.from_config(self.pipe.scheduler.config)
        print(f"Scheduler: {self.pipe.scheduler.__class__}")

        self.pipe = self.pipe.to(device)
        print(f"Device: {self.pipe.device}")

        if torch.cuda.is_available() and torch.__version__ < "2.0":
            print("Enabled: Xformers memory efficient attention")
            self.pipe.enable_xformers_memory_efficient_attention()

        # Add extra choices to self.wid_scheduler
        self.w_scheduler.options = self.pipe.scheduler.compatibles
        # Set the current scheduler to the default
        self.w_scheduler.value = self.pipe.scheduler.__class__

        # Update the signature
        self.__pipe_signature = hash((model, self.pipe.scheduler.__class__, device))

    def __call__(
        self,
        model=None,
        scheduler=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=None,
        interactive=False,
        return_time_elapsed=False,
        **kwargs,
    ):
        # Parameter overrides
        model = model or self.model
        scheduler = scheduler or self.scheduler_cls
        seed = seed or self.seed

        clear_output()

        # Initialize the pipeline
        self.init_pipeline(model, scheduler, device)

        # Create generator if a seed is provided
        _generator = None
        if seed is not None and seed != -1:
            print(f"Seed: {seed}")
            _generator = torch.Generator().manual_seed(seed)

        # Parameters passed to the pipeline's __call__ method
        pipe_call_params = {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "num_images_per_prompt": self.num_images_per_prompt,
            "generator": _generator,
        }
        # Any overrides passed to the function in kwargs
        pipe_call_params.update(kwargs)
        print(f"{self.pipe.__class__.__name__}.__call__ parameters:")
        pprint(pipe_call_params, sort_dicts=False)

        time_start = time.process_time()
        # Run the pipeline
        out = self.pipe(**pipe_call_params)
        time_end = time.process_time()
        time_elapsed = time_end - time_start

        # In interactive mode display the controls again and plot the images
        if interactive:
            self.show()
            helpers.plot(out.images, save_fname="plot_latest_generation")

        return (out.images, time_elapsed) if return_time_elapsed else out.images
