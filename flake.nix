{
  description = "Zola development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            zola
            # Additional useful tools for web development
            # nodePackages.live-server
            # html-tidy
          ];

          shellHook = ''
            echo "Zola development environment activated!"
            echo "Available commands:"
            echo "  zola serve  - Start development server"
            echo "  zola build  - Build site"
            echo "  zola init   - Create new site"
          '';
        };
      }
    );
}
