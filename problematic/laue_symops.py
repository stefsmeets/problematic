symops = {
    '1': [
        'x, y, z'
    ],

    '-1': [
        'x, y, z',
        '-x, -y, -z'
    ],

    '2/m:a': [
        'x, y, z',
        '-x, y, -z',
        '-x, -y, -z',
        'x, -y, z'
    ],

    '2/m:b': [
        'x, y, z',
        '-x, y, -z',
        '-x, -y, -z',
        'x, -y, z'
    ],

    '2/m:c': [
        'x, y, z',
        '-x, -y, z',
        '-x, -y, -z',
        'x, y, -z'
    ],

    'mmm': [
        'x, y, z',
        '-x, -y, z',
        'x, -y, -z',
        '-x, y, -z',
        '-x, -y, -z',
        'x, y, -z',
        '-x, y, z',
        'x, -y, z'
    ],

    '4/m': [
        'x, y, z',
        '-y, x, z',
        '-x, -y, z',
        'y, -x, z',
        '-x, -y, -z',
        'y, -x, -z',
        'x, y, -z',
        '-y, x, -z'
    ],

    '4/mmm': [
        'x, y, z',
        '-y, x, z',
        '-x, -y, z',
        'y, -x, z',
        'x, -y, -z',
        '-x, y, -z',
        'y, x, -z',
        '-y, -x, -z',
        '-x, -y, -z',
        'y, -x, -z',
        'x, y, -z',
        '-y, x, -z',
        '-x, y, z',
        'x, -y, z',
        '-y, -x, z',
        'y, x, z'
    ],

    '-3': [
        'x, y, z',
        '-y, x-y, z',
        '-x+y, -x, z',
        '-x, -y, -z',
        'y, -x+y, -z',
        'x-y, x, -z'
    ],

    '-3m': [
        'x, y, z',
        '-y, x-y, z',
        '-x+y, -x, z',
        'x-y, -y, -z',
        '-x, -x+y, -z',
        'y, x, -z',
        '-x, -y, -z',
        'y, -x+y, -z',
        'x-y, x, -z',
        '-x+y, y, z',
        'x, x-y, z',
        '-y, -x, z'
    ],

    '-3m1': [
        'x, y, z',
        '-y, x-y, z',
        '-x+y, -x, z',
        'x-y, -y, -z',
        '-x, -x+y, -z',
        'y, x, -z',
        '-x, -y, -z',
        'y, -x+y, -z',
        'x-y, x, -z',
        '-x+y, y, z',
        'x, x-y, z',
        '-y, -x, z'
    ],

    '-31m': [
        'x, y, z',
        '-y, x-y, z',
        '-x+y, -x, z',
        '-y, -x, -z',
        '-x+y, y, -z',
        'x, x-y, -z',
        '-x, -y, -z',
        'y, -x+y, -z',
        'x-y, x, -z',
        'y, x, z',
        'x-y, -y, z',
        '-x, -x+y, z'
    ],

    '6/m': [
        'x, y, z',
        'x-y, x, z',
        '-y, x-y, z',
        '-x, -y, z',
        '-x+y, -x, z',
        'y, -x+y, z',
        '-x, -y, -z',
        '-x+y, -x, -z',
        'y, -x+y, -z',
        'x, y, -z',
        'x-y, x, -z',
        '-y, x-y, -z'
    ],

    '6/mmm': [
        'x, y, z',
        'x-y, x, z',
        '-y, x-y, z',
        '-x, -y, z',
        '-x+y, -x, z',
        'y, -x+y, z',
        'x-y, -y, -z',
        '-x, -x+y, -z',
        'y, x, -z',
        '-y, -x, -z',
        '-x+y, y, -z',
        'x, x-y, -z',
        '-x, -y, -z',
        '-x+y, -x, -z',
        'y, -x+y, -z',
        'x, y, -z',
        'x-y, x, -z',
        '-y, x-y, -z',
        '-x+y, y, z',
        'x, x-y, z',
        '-y, -x, z',
        'y, x, z',
        'x-y, -y, z',
        '-x, -x+y, z'
    ],

    'm-3': [
        'x, y, z',
        'z, x, y',
        'y, z, x',
        '-y, -z, x',
        'z, -x, -y',
        '-y, z, -x',
        '-z, -x, y',
        '-z, x, -y',
        'y, -z, -x',
        '-x, -y, z',
        'x, -y, -z',
        '-x, y, -z',
        '-x, -y, -z',
        '-z, -x, -y',
        '-y, -z, -x',
        'y, z, -x',
        '-z, x, y',
        'y, -z, x',
        'z, x, -y',
        'z, -x, y',
        '-y, z, x',
        'x, y, -z',
        '-x, y, z',
        'x, -y, z'
    ],

    'm-3m': [
        'x, y, z',
        '-y, x, z',
        '-x, -y, z',
        'y, -x, z',
        'x, -z, y',
        'x, -y, -z',
        'x, z, -y',
        'z, y, -x',
        '-x, y, -z',
        '-z, y, x',
        'z, x, y',
        'y, z, x',
        '-y, -z, x',
        'z, -x, -y',
        '-y, z, -x',
        '-z, -x, y',
        '-z, x, -y',
        'y, -z, -x',
        'y, x, -z',
        '-y, -x, -z',
        '-x, z, y',
        '-x, -z, -y',
        'z, -y, x',
        '-z, -y, -x',
        '-x, -y, -z',
        'y, -x, -z',
        'x, y, -z',
        '-y, x, -z',
        '-x, z, -y',
        '-x, y, z',
        '-x, -z, y',
        '-z, -y, x',
        'x, -y, z',
        'z, -y, -x',
        '-z, -x, -y',
        '-y, -z, -x',
        'y, z, -x',
        '-z, x, y',
        'y, -z, x',
        'z, x, -y',
        'z, -x, y',
        '-y, z, x',
        '-y, -x, z',
        'y, x, z',
        'x, -z, -y',
        'x, z, y',
        '-z, y, -x',
        'z, y, x'
    ]
}
