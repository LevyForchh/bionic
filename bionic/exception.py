'''
Bionic-specific exception classes.
'''


class UndefinedEntityError(KeyError):
    @classmethod
    def for_name(cls, name):
        return cls("Entity %r is not defined" % name)


class AlreadyDefinedEntityError(ValueError):
    @classmethod
    def for_name(cls, name):
        return cls("Entity %r is already defined" % name)
