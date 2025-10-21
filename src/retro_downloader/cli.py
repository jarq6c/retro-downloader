"""Command-line interface."""
import click

@click.command()
def cli() -> None:
    """Primary entry-point."""
    print("CLI")

if __name__ == "__main__":
    cli()
