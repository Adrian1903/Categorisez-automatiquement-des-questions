
function tagger(lst, id, style) {
    lst.forEach(element => {
        var newBut = document.createElement("button");
        var newButContent = document.createTextNode(element);
        newBut.appendChild(newButContent);
        newBut.setAttribute("class", style)

        var currentDiv = document.getElementById(id);
        document.body.insertBefore(newBut, currentDiv.nextSibling);
    });
}

function H2(title, parent_id, id) {
    var newH2 = document.createElement("h2");
    var newH2Content = document.createTextNode(title);
    newH2.appendChild(newH2Content);
    newH2.setAttribute("id", id)

    var currentDiv = document.getElementById(parent_id);
    document.body.insertBefore(newH2, currentDiv.nextSibling);

}
