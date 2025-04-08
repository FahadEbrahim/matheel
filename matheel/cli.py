import click
from .similarity import get_sim_list

@click.group()
def main():
    """Matheel CLI - Compute Code Similarity"""
    pass

@main.command()
@click.argument('zipfile', type=click.Path(exists=True))
@click.option('--ws', default=0.7, help='Semantic Similarity Weight')
@click.option('--wl', default=0.3, help='Levenshtein Distance Weight')
@click.option('--wj', default=0.0, help='Jaro-Winkler Distance Weight')
@click.option('--model', default='uclanlp/plbart-java-cs', help='Sentence Transformer Model')
@click.option('--threshold', default=0.0, help='Similarity Threshold')
@click.option('--num', default=10, help='Number of Results to Display')
def compare(zipfile, ws, wl, wj, model, pooling, threshold, num):
    """Bulk similarity from ZIP file"""
    results = get_sim_list(zipfile, ws, wl, wj, model, threshold, num)
    click.echo(results.to_string(index=False))
