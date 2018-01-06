/****************************************************************************
** Meta object code from reading C++ file 'streamserver.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.7.0)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../src/streamserver.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#include <QtCore/QList>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'streamserver.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.7.0. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
struct qt_meta_stringdata_StreamServer_t {
    QByteArrayData data[12];
    char stringdata0[150];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_StreamServer_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_StreamServer_t qt_meta_stringdata_StreamServer = {
    {
QT_MOC_LITERAL(0, 0, 12), // "StreamServer"
QT_MOC_LITERAL(1, 13, 6), // "closed"
QT_MOC_LITERAL(2, 20, 0), // ""
QT_MOC_LITERAL(3, 21, 15), // "onNewConnection"
QT_MOC_LITERAL(4, 37, 11), // "onSslErrors"
QT_MOC_LITERAL(5, 49, 16), // "QList<QSslError>"
QT_MOC_LITERAL(6, 66, 6), // "errors"
QT_MOC_LITERAL(7, 73, 18), // "processTextMessage"
QT_MOC_LITERAL(8, 92, 7), // "message"
QT_MOC_LITERAL(9, 100, 20), // "processBinaryMessage"
QT_MOC_LITERAL(10, 121, 9), // "createKDE"
QT_MOC_LITERAL(11, 131, 18) // "socketDisconnected"

    },
    "StreamServer\0closed\0\0onNewConnection\0"
    "onSslErrors\0QList<QSslError>\0errors\0"
    "processTextMessage\0message\0"
    "processBinaryMessage\0createKDE\0"
    "socketDisconnected"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_StreamServer[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: name, argc, parameters, tag, flags
       1,    0,   49,    2, 0x06 /* Public */,

 // slots: name, argc, parameters, tag, flags
       3,    0,   50,    2, 0x08 /* Private */,
       4,    1,   51,    2, 0x08 /* Private */,
       7,    1,   54,    2, 0x08 /* Private */,
       9,    1,   57,    2, 0x08 /* Private */,
      10,    0,   60,    2, 0x08 /* Private */,
      11,    0,   61,    2, 0x08 /* Private */,

 // signals: parameters
    QMetaType::Void,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, 0x80000000 | 5,    6,
    QMetaType::Void, QMetaType::QString,    8,
    QMetaType::Void, QMetaType::QByteArray,    8,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void StreamServer::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        StreamServer *_t = static_cast<StreamServer *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->closed(); break;
        case 1: _t->onNewConnection(); break;
        case 2: _t->onSslErrors((*reinterpret_cast< const QList<QSslError>(*)>(_a[1]))); break;
        case 3: _t->processTextMessage((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 4: _t->processBinaryMessage((*reinterpret_cast< QByteArray(*)>(_a[1]))); break;
        case 5: _t->createKDE(); break;
        case 6: _t->socketDisconnected(); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 2:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QList<QSslError> >(); break;
            }
            break;
        }
    } else if (_c == QMetaObject::IndexOfMethod) {
        int *result = reinterpret_cast<int *>(_a[0]);
        void **func = reinterpret_cast<void **>(_a[1]);
        {
            typedef void (StreamServer::*_t)();
            if (*reinterpret_cast<_t *>(func) == static_cast<_t>(&StreamServer::closed)) {
                *result = 0;
                return;
            }
        }
    }
}

const QMetaObject StreamServer::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_StreamServer.data,
      qt_meta_data_StreamServer,  qt_static_metacall, Q_NULLPTR, Q_NULLPTR}
};


const QMetaObject *StreamServer::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *StreamServer::qt_metacast(const char *_clname)
{
    if (!_clname) return Q_NULLPTR;
    if (!strcmp(_clname, qt_meta_stringdata_StreamServer.stringdata0))
        return static_cast<void*>(const_cast< StreamServer*>(this));
    return QObject::qt_metacast(_clname);
}

int StreamServer::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 7)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 7;
    }
    return _id;
}

// SIGNAL 0
void StreamServer::closed()
{
    QMetaObject::activate(this, &staticMetaObject, 0, Q_NULLPTR);
}
QT_END_MOC_NAMESPACE
