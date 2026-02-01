const fs = require('fs');

const data = fs.readFileSync('README.md', 'utf8');

const newData = data.replace(/React/ig, 'Angular'); // Modificamos todo lo que se hacía con React para hacerlo con Angular, una paquetería de Js


fs.writeFileSync('README-Angular.md', newData);

