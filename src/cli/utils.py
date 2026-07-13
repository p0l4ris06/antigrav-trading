import click

def format_banner(text: str):
    """Format and print a CLI banner."""
    click.echo("=" * 60)
    click.echo(text.center(60))
    click.echo("=" * 60)
