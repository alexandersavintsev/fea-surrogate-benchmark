from src.validate_physics import (
    check_elastic_cantilever,
    check_kirsch,
    check_plasticity,
    check_resonance,
)


def test_physical_checks_pass():
    checks = [
        check_kirsch(),
        check_elastic_cantilever(),
        check_plasticity(),
        check_resonance(),
    ]
    assert all(check["Пройдена"] for check in checks)
