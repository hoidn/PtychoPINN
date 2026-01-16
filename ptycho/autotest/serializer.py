# spec
#module DebuggingSystem {
#
#    interface Serializer {
#        """
#        Serializes Python objects to a binary format using pickle.
#
#        Preconditions:
#        - `input_data` must be a picklable Python object.
#
#        Postconditions:
#        - Returns the serialized binary data of the input object.
#        - Raises ValueError if the input data is not picklable.
#        """
#        bytes serialize(Any input_data);
#
#        """
#        Deserializes Python objects from a binary format using pickle.
#
#        Preconditions:
#        - `serialized_data` must be a valid pickle-serialized binary string.
#
#        Postconditions:
#        - Returns the deserialized Python object.
#        - Raises ValueError if the binary data could not be deserialized.
#        """
#        Any deserialize(bytes serialized_data);
#    };

import doctest
import pickle
from typing import Any, List

class Serializer:
    def serialize(self, input_data: Any) -> bytes:
        """
        Serializes Python objects to a binary format using pickle.

        Preconditions:
        - `input_data` must be a picklable Python object.

        Postconditions:
        - Returns the serialized binary data of the input object.
        - Raises ValueError if the input data is not picklable.

        >>> s = Serializer()
        >>> data = {'key': 'value'}
        >>> serialized_data = s.serialize(data)
        >>> type(serialized_data)
        <class 'bytes'>
        >>> deserialized_data = s.deserialize(serialized_data)
        >>> deserialized_data == data
        True
        >>> s.serialize(lambda x: x)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Input data is not picklable
        >>> s.deserialize(b'not a pickle')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Could not deserialize the binary data
        """
        try:
            return pickle.dumps(input_data)
        except (pickle.PicklingError, AttributeError, TypeError):
            raise ValueError("Input data is not picklable")

    def deserialize(self, serialized_data: bytes) -> Any:
        """
        Deserializes Python objects from a binary format using pickle.

        Preconditions:
        - `serialized_data` must be a valid pickle-serialized binary string.

        Postconditions:
        - Returns the deserialized Python object.
        - Raises ValueError if the binary data could not be deserialized.

        >>> s = Serializer()
        >>> data = {'key': 'value'}
        >>> serialized_data = s.serialize(data)
        >>> deserialized_data = s.deserialize(serialized_data)
        >>> deserialized_data == data
        True
        >>> s.deserialize(b'not a pickle')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ValueError: Could not deserialize the binary data
        """
        try:
            return pickle.loads(serialized_data)
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError):
            raise ValueError("Could not deserialize the binary data")
# doctest.testmod(verbose=True)  # Commented out - should not run at import time

