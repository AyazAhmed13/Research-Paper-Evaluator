"""
run.py
------
CLI runner for quick testing without the Streamlit UI.
Usage:
    python run.py https://arxiv.org/abs/1706.03762
    python run.py https://arxiv.org/abs/1706.03762 --no-pdf
"""

import sys
import argparse
from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Agentic Research Paper Evaluator — CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py https://arxiv.org/abs/1706.03762
  python run.py https://arxiv.org/abs/1810.04805 --no-pdf
        """
    )
    parser.add_argument("url",     help="arXiv paper URL")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF report generation")
    parser.add_argument("--no-md",  action="store_true", help="Skip Markdown report generation")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  Agentic Research Paper Evaluator")
    print(f"{'='*60}")
    print(f"  Paper URL: {args.url}")
    print(f"{'='*60}\n")

    try:
        from crew import run_evaluation
        from report_generator import save_markdown_report, generate_pdf_report

        # Run evaluation
        evaluation = run_evaluation(args.url)

        # Print score summary
        scores = evaluation["scores"]
        print(f"\n{'='*60}")
        print("  EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"  Paper: {evaluation['paper_meta']['title']}")
        print(f"  Verdict:             {scores.get('overall_verdict', 'N/A')}")
        print(f"  Consistency Score:   {scores.get('consistency_score', 'N/A')}/100")
        print(f"  Grammar Rating:      {scores.get('grammar_rating', 'N/A')}")
        print(f"  Novelty Index:       {scores.get('novelty_index', 'N/A')}")
        print(f"  Fabrication Risk:    {scores.get('fabrication_probability', 'N/A')}%")
        fc = scores.get("fact_check_summary", {})
        print(f"  Facts Verified:      {fc.get('verified_count', 'N/A')}")
        print(f"  Facts Contradicted:  {fc.get('contradicted_count', 'N/A')}")
        print(f"{'='*60}\n")

        # Generate reports
        if not args.no_md:
            md_path = save_markdown_report(evaluation)
            print(f"Markdown report: {md_path}")

        if not args.no_pdf:
            pdf_path = generate_pdf_report(evaluation)
            print(f"PDF report:      {pdf_path}")

        print("\nDone!")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()