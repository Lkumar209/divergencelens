"""CLI entry point for DivergenceLens."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="divergencelens",
    help="Silent divergence auditing for LangChain Deep Agents.",
    add_completion=False,
)
console = Console()


@app.command()
def audit(
    target: str = typer.Argument(..., help="LangSmith run ID or path to a trace JSON file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save report JSON to this path"),
    judge: bool = typer.Option(False, "--judge", help="Enable LLM judge (requires API key)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Audit a single run for silent divergence."""
    from divergencelens.core.config import DivergenceLensConfig, DetectionConfig
    from divergencelens.sdk.client import DivergenceLens

    config = DivergenceLensConfig(
        detection=DetectionConfig(enable_judge=judge)
    )
    lens = DivergenceLens(config)

    with console.status("Auditing run..."):
        if Path(target).exists():
            result = lens.audit_json(target)
        else:
            result = lens.audit_langsmith_run(target)

    # Print summary table
    table = Table(title=f"DivergenceLens — Run {result.run_id[:12]}…")
    table.add_column("Category", style="cyan")
    table.add_column("Severity", style="yellow")
    table.add_column("Confidence", style="green")
    table.add_column("Rationale")

    for div in result.divergences:
        table.add_row(
            div.category.value,
            div.severity.value,
            f"{div.confidence:.2f}",
            div.rationale[:80],
        )

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] {result.summary}")
    console.print(f"[dim]Audit took {result.duration_ms:.0f}ms[/dim]")

    if output:
        Path(output).write_text(result.model_dump_json(indent=2))
        console.print(f"Report saved to [green]{output}[/green]")

    if not result.is_clean:
        sys.exit(1)


@app.command()
def bench(
    split: str = typer.Option("test", "--split", help="Dataset split: train|dev|test"),
    output_dir: str = typer.Option("results/", "--output-dir", "-o"),
    seeds: int = typer.Option(3, "--seeds"),
    judge: bool = typer.Option(False, "--judge"),
) -> None:
    """Run the DivergenceBench benchmark and emit RESULTS.md."""
    console.print(f"[bold]Running DivergenceBench[/bold] (split={split}, seeds={seeds})")
    try:
        from bench.metrics.compute import run_benchmark
        results = run_benchmark(split=split, n_seeds=seeds, enable_judge=judge, output_dir=output_dir)
        console.print(f"[green]Done.[/green] Results written to {output_dir}")
        console.print(results.get("summary", {}))
    except ImportError as exc:
        console.print(f"[red]bench module not available:[/red] {exc}")
        raise typer.Exit(1)


@app.command()
def report(
    run_id: str = typer.Argument(..., help="LangSmith run ID or path to audit result JSON"),
    format: str = typer.Option("markdown", "--format", help="Output format: markdown|json"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
) -> None:
    """Generate a divergence report for a run."""
    from divergencelens.report.run_report import RunReporter
    from divergencelens.sdk.client import DivergenceLens, AuditResult

    if Path(run_id).exists():
        result = AuditResult(**json.loads(Path(run_id).read_text()))
    else:
        lens = DivergenceLens()
        result = lens.audit_langsmith_run(run_id)

    reporter = RunReporter()
    report_obj = reporter.generate_from_result(result)

    if format == "json":
        content = report_obj.model_dump_json(indent=2)
    else:
        content = report_obj.markdown

    if output:
        Path(output).write_text(content)
        console.print(f"Report saved to [green]{output}[/green]")
    else:
        console.print(content)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the DivergenceLens audit service."""
    import uvicorn
    console.print(f"Starting DivergenceLens serve on http://{host}:{port}")
    uvicorn.run(
        "divergencelens.serve.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
