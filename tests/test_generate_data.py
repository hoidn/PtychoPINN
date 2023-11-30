# Test for generate_data module in the ptycho package

from ptycho import generate_data as init

def test_placeholder():
    # Placeholder test to ensure the import works
    assert hasattr(init, 'PtychoData'), "generate_data module should have PtychoData class"
