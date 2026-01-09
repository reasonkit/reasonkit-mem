# Contributing to ReasonKit Mem

Thank you for your interest in contributing to ReasonKit Mem!

## Quick Start

```bash
# Clone the repository
git clone https://github.com/reasonkit/reasonkit-mem
cd reasonkit-mem

# Run tests
cargo test

# Run clippy
cargo clippy -- -D warnings

# Format code
cargo fmt
```

## The 5 Gates of Quality

All contributions must pass:

1. **Build**: `cargo build --release`
2. **Lint**: `cargo clippy -- -D warnings`
3. **Format**: `cargo fmt --check`
4. **Test**: `cargo test --all-features`
5. **Bench**: `cargo bench` (no regression > 5%)

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Ensure all 5 gates pass
5. Commit with a clear message
6. Push and open a PR

## Code Style

- Follow Rust conventions
- Document public APIs
- Add tests for new functionality
- Keep PRs focused and small

## Questions?

Open an issue or reach out at [team@reasonkit.sh](mailto:team@reasonkit.sh).

---

_Part of the ReasonKit Ecosystem_
