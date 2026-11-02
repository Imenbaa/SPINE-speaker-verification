from pytorch_lightning.cli import LightningCLI


def cli_main():
    LightningCLI(
            save_config_kwargs={"multifile": True},
            parser_kwargs={"parser_mode": "omegaconf"},
            )


if __name__ == '__main__':
    cli_main()
