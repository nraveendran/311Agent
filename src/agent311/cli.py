"""Command line entrypoint for the 311 routing agent."""

from __future__ import annotations

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from agent311.config import load_config
from agent311.orchestrator import classify_ticket


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=None,
        help="Path to a properties file. Defaults to ./config.properties or AGENT311_CONFIG_PATH.",
    )
    known_args, _ = config_parser.parse_known_args()
    config = load_config(known_args.config)

    parser = argparse.ArgumentParser(
        description="Classify a 311 ticket into the correct department using MCP tools.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--description",
        required=True,
        help="Citizen complaint description from the 311 ticket.",
    )
    parser.add_argument(
        "--address",
        required=True,
        help="Address associated with the 311 ticket.",
    )
    parser.add_argument(
        "--mcp-url",
        default=None,
        help=f"Streamable HTTP MCP server URL. Defaults to config value: {config.mcp_url}",
    )
    parser.add_argument(
        "--mcp-transport",
        default=None,
        choices=["streamable_http", "http", "sse"],
        help=(
            "MCP transport label expected by langchain-mcp-adapters. "
            f"Defaults to config value: {config.mcp_transport}"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenAI chat model used by the orchestrator. Defaults to config value: {config.model}",
    )
    parser.add_argument(
        "--classification-tool-name",
        default=None,
        help=(
            "Required MCP classification tool name. "
            f"Defaults to config value: {config.classification_tool_name or 'any MCP tool'}"
        ),
    )
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=f"Console log level. Defaults to config value: {config.log_level}",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)
    configure_logging(args.log_level or config.log_level)
    decision = asyncio.run(
        classify_ticket(
            description=args.description,
            address=args.address,
            mcp_url=args.mcp_url,
            mcp_transport=args.mcp_transport,
            model=args.model,
            classification_tool_name=args.classification_tool_name,
            config_path=args.config,
        )
    )
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
