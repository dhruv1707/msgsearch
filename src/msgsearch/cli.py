"""Command line interface.

All argument parsing lives here so the other modules stay importable libraries
with no opinion about how they are invoked. One command with subcommands, rather
than a directory of scripts, because it gives people a single thing to remember
and a single place to find help:

    msgsearch doctor               check this machine is set up correctly
    msgsearch explore              what is in your database
    msgsearch index                build the search index
    msgsearch search "a query"     search it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, config


def _add_index_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "index",
        help="build the search index",
        description="Read the message database, group messages into conversation "
        "windows, embed the passages inside them, and write the index.",
    )
    parser.add_argument(
        "--chat",
        default=config.TESTBED_CHAT,
        metavar="ID",
        help="restrict to one conversation, by phone number or email "
        "(default: $MSGSEARCH_TESTBED_CHAT, or all conversations)",
    )
    parser.add_argument("--index-dir", metavar="DIR", help="where to write the index")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="re-embed everything instead of reusing unchanged passages",
    )
    parser.set_defaults(handler=_run_index)


def _run_index(args) -> int:
    from .index import build

    build(
        chat_identifier=args.chat,
        index_dir=args.index_dir,
        batch_size=args.batch_size,
        rebuild=args.rebuild,
    )
    return 0


def _add_search_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "search",
        help="search the index",
        description="Find conversations by meaning as well as by keyword.",
    )
    parser.add_argument("query", nargs="+", help="what you are looking for")
    parser.add_argument("--limit", type=int, default=config.DEFAULT_LIMIT, metavar="N")
    parser.add_argument(
        "--chat", metavar="TEXT", help="restrict to matching conversations"
    )
    parser.add_argument(
        "--from", dest="from_", metavar="WHO", help="restrict to a speaker"
    )
    parser.add_argument("--after", metavar="YYYY-MM-DD")
    parser.add_argument("--before", metavar="YYYY-MM-DD")
    parser.add_argument(
        "--type",
        metavar="TAG",
        help="restrict by shape: credential, credential_talk, email, phone, url, address",
    )
    parser.add_argument("--index-dir", metavar="DIR")
    parser.add_argument("--snippet", type=int, default=1200, metavar="CHARS")
    parser.add_argument(
        "--full", action="store_true", help="show the whole window, not just the match"
    )
    parser.add_argument(
        "--rerank",
        dest="no_rerank",
        action="store_false",
        default=not config.RERANK_ENABLED,
        help="run the cross-encoder (off by default; it measurably hurts)",
    )
    parser.add_argument("--no-rerank", dest="no_rerank", action="store_true")
    parser.add_argument("--no-dense", action="store_true", help="keyword search only")
    parser.add_argument("--no-bm25", action="store_true", help="vector search only")
    parser.set_defaults(handler=_run_search)


def _run_search(args) -> int:
    from .search import format_result, search

    query = " ".join(args.query)
    try:
        results = search(query, args)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 1

    if not results:
        print("no results")
        return 0

    print(f"{len(results)} result(s) for {query!r}")
    for position, result in enumerate(results, start=1):
        print(format_result(position, result, args.snippet, full=args.full))
    return 0


def _add_contacts_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "contacts",
        help="map phone numbers and emails to names",
        description="Speakers appear by name rather than by phone number, which "
        "reads better and embeds better. Names come from an alias file you "
        "control; this command creates and inspects it.",
    )
    parser.add_argument(
        "--import",
        dest="import_path",
        metavar="FILE.vcf",
        help="merge names from a vCard export (Contacts.app: File -> Export)",
    )
    parser.add_argument(
        "--from-addressbook",
        action="store_true",
        help="merge names from macOS Contacts (needs Contacts permission)",
    )
    parser.add_argument(
        "--template",
        type=int,
        nargs="?",
        const=25,
        metavar="N",
        help="write an alias file stub for the N busiest unnamed handles, for you "
        "to fill in (default 25)",
    )
    parser.add_argument(
        "--unresolved",
        type=int,
        nargs="?",
        const=25,
        metavar="N",
        help="list the N busiest handles that still have no name (default 25)",
    )
    parser.set_defaults(handler=_run_contacts)


def _run_contacts(args) -> int:
    from . import contacts as contacts_module
    from .extract import connect

    path = contacts_module.aliases_path()
    mapping = contacts_module.load_mapping()

    imported = {}
    if args.import_path:
        imported = contacts_module.read_vcard_file(Path(args.import_path))
        print(f"read {len(imported):,} handles from {args.import_path}")
    elif args.from_addressbook:
        try:
            imported = contacts_module.read_addressbook()
        except (PermissionError, FileNotFoundError) as error:
            print(error, file=sys.stderr)
            return 1
        print(f"read {len(imported):,} handles from macOS Contacts")

    if imported:
        added = {k: v for k, v in imported.items() if k not in mapping}
        mapping.update(imported)
        contacts_module.save(mapping, path)
        print(f"wrote {len(mapping):,} names to {path} ({len(added):,} new)")

    # Report each source separately, because "no names" has two very different
    # causes: the Contacts permission is not granted, or it is granted and simply
    # has no entry for these handles.
    try:
        from_contacts = contacts_module.read_addressbook()
        contacts_status = f"{len(from_contacts):,} entries"
    except PermissionError:
        from_contacts = {}
        contacts_status = (
            "not permitted — grant Contacts access to the app running msgsearch "
            "under System Settings > Privacy & Security > Contacts"
        )
    except (FileNotFoundError, OSError) as error:
        from_contacts = {}
        contacts_status = str(error)

    lookup = contacts_module.Contacts.load()
    db = connect()
    try:
        volumes = contacts_module.handle_volumes(db)
    finally:
        db.close()

    named = [(h, n) for h, n in volumes if lookup.name(h)]
    unnamed = [(h, n) for h, n in volumes if not lookup.name(h)]
    covered = sum(n for _, n in named)
    total = sum(n for _, n in volumes) or 1

    overrides = {k: v for k, v in mapping.items() if v}
    print(f"\nmacOS Contacts : {contacts_status}")
    print(f"alias file     : {len(overrides):,} names in {path}")
    print(
        f"resolved       : {len(named):,} of {len(volumes):,} handles, "
        f"covering {100 * covered / total:.1f}% of received messages"
    )

    if args.template:
        stub = dict(mapping)
        for handle, _ in unnamed[: args.template]:
            stub.setdefault(handle, "")
        contacts_module.save(stub, path)
        print(
            f"\nwrote a stub for {min(args.template, len(unnamed))} handles to {path}"
            "\nFill in the names and re-run 'msgsearch index'. Entries left empty "
            "are ignored."
        )
        return 0

    if args.unresolved:
        print(f"\nbusiest handles with no name (top {args.unresolved}):")
        for handle, count in unnamed[: args.unresolved]:
            print(f"  {count:>7,}  {handle}")
        print(f"\nAdd them to {path} as a JSON object of handle -> name.")
    return 0


def _add_doctor_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="check that this machine is set up correctly",
        description="Verify the interpreter, PyTorch, the message database, the "
        "embedding model and the index, and say how to fix whatever is wrong. "
        "Prints no message content.",
    )
    parser.set_defaults(handler=_run_doctor)


def _run_doctor(args) -> int:
    from .doctor import run

    return run()


def _add_explore_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "explore",
        help="report what is in a message database",
        description="Structural reconnaissance: message counts, how many rows hide "
        "their text in attributedBody, the date range, and the busiest chats. "
        "Prints no message content.",
    )
    parser.set_defaults(handler=_run_explore)


def _run_explore(args) -> int:
    from .explore import main as explore_main

    explore_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msgsearch",
        description="Search your iMessage history by meaning, entirely on your "
        "own machine.",
        epilog="Run 'msgsearch <command> --help' for details of a command.",
    )
    parser.add_argument("--version", action="version", version=f"msgsearch {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    _add_contacts_command(subparsers)
    _add_doctor_command(subparsers)
    _add_explore_command(subparsers)
    _add_index_command(subparsers)
    _add_search_command(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 1
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
