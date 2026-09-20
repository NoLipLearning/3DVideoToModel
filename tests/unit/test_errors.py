"""V2MError: message + remedy formatting."""

from v2m.errors import ConfigError, V2MError


def test_v2m_error_includes_remedy_in_str():
    err = V2MError("something broke", remedy="try again")
    assert "something broke" in str(err)
    assert "try again" in str(err)
    assert err.message == "something broke"
    assert err.remedy == "try again"


def test_v2m_error_without_remedy():
    err = V2MError("no remedy given")
    assert str(err) == "no remedy given"
    assert err.remedy is None


def test_subclasses_are_v2m_errors():
    assert issubclass(ConfigError, V2MError)
