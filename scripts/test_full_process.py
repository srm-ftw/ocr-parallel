#!/usr/bin/env python3
"""
Script de prueba para el Full Process Pipeline.

Uso:
    # Modo local con CSV
    python scripts/test_full_process.py ocr/scripts/escritura_test_2.pdf --csv ocr/scripts/process_test.csv
    
    # Modo GCS con CSV
    python scripts/test_full_process.py --bucket bucket-escrituras-smr --file escritura_test.pdf --csv inmuebles.csv
    
    # Sin argumentos (usa default local, inmueble de prueba si no hay CSV)
    python scripts/test_full_process.py

Formato CSV esperado:
    Inmueble,Folio
    "APARTAMENTO 102 -T3","294-109668"
    "Casa Lote...","370-404755"
    
    Nota: El CSV debe tener columnas "Inmueble" y "Folio" (case-insensitive).
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.full_extractor_service import FullExtractorService
from app.services.full_ocr_service import FullOCRService
from app.models.full_process import InmuebleInput


def load_inmuebles_from_csv(csv_path: str) -> List[InmuebleInput]:
    """
    Load inmuebles from CSV file.
    
    Expected CSV format:
        Inmueble,Folio
        "APARTAMENTO 102 -T3","294-109668"
        "Casa Lote...","370-404755"
    
    Args:
        csv_path: Path to CSV file (always treated as relative to script directory)
    
    Returns:
        List of InmuebleInput objects
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    # Always resolve relative to script directory
    csv_file = Path(__file__).parent / csv_path
    
    if not csv_file.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_file}\n"
            f"  (resolved from: {csv_path} relative to {Path(__file__).parent})"
        )
    
    inmuebles = []
    
    with open(csv_file, "r", encoding="utf-8") as f:
        # Try to detect delimiter
        sample = f.read(1024)
        f.seek(0)
        sniffer = csv.Sniffer()
        delimiter = sniffer.sniff(sample).delimiter
        
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Normalize column names (case-insensitive, strip whitespace)
        fieldnames = {name.strip().lower(): name for name in reader.fieldnames or []}
        
        if "inmueble" not in fieldnames or "folio" not in fieldnames:
            raise ValueError(
                f"CSV must contain 'Inmueble' and 'Folio' columns. "
                f"Found columns: {list(reader.fieldnames or [])}"
            )
        
        for row_num, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
            inmueble_col = fieldnames["inmueble"]
            folio_col = fieldnames["folio"]
            
            inmueble = row.get(inmueble_col, "").strip()
            folio = row.get(folio_col, "").strip()
            
            if not inmueble:
                print(f"  ⚠ Warning: Row {row_num} has empty 'Inmueble', skipping...")
                continue
            
            if not folio:
                print(f"  ⚠ Warning: Row {row_num} has empty 'Folio', skipping...")
                continue
            
            inmuebles.append(
                InmuebleInput(
                    inmueble=inmueble,
                    folio=folio
                )
            )
    
    if not inmuebles:
        raise ValueError("No valid inmuebles found in CSV file")
    
    return inmuebles


async def test_full_process(
    local_path: Optional[str] = None,
    bucket: Optional[str] = None,
    file_name: Optional[str] = None,
    csv_path: Optional[str] = None
):
    """
    Test full process with either local file or GCS.
    
    Args:
        local_path: Path to local PDF file (relative or absolute)
        bucket: GCS bucket name
        file_name: GCS file name
        csv_path: Path to CSV file with inmuebles (relative or absolute)
    """
    print("=" * 80)
    print("TEST: Full Process Pipeline")
    print("=" * 80)

    # Determine mode
    if local_path:
        # Local file mode - always resolve relative to script directory
        test_pdf_path = Path(local_path)
        if not test_pdf_path.is_absolute():
            test_pdf_path = Path(__file__).parent / local_path

        if not test_pdf_path.exists():
            print(f"ERROR: Test PDF not found at {test_pdf_path}")
            print(f"  (resolved from: {local_path} relative to {Path(__file__).parent})")
            return

        print(f"\n1. Mode: LOCAL FILE")
        print(f"   Path: {test_pdf_path}")
        print(f"   Size: {test_pdf_path.stat().st_size / 1024:.2f} KB")
        use_local = True
        display_name = test_pdf_path.name

    elif bucket and file_name:
        # GCS mode
        print(f"\n1. Mode: GCS")
        print(f"   Bucket: {bucket}")
        print(f"   File: {file_name}")
        use_local = False
        display_name = file_name

    else:
        print("ERROR: Must provide either:")
        print("  - Local file path: python scripts/test_full_process.py escritura_test.pdf")
        print("    (paths are relative to scripts/ directory)")
        print("  - GCS: python scripts/test_full_process.py --bucket BUCKET --file FILE")
        return

    # Initialize services
    print("\n2. Initializing services...")
    full_ocr_service = FullOCRService()
    full_extractor_service = FullExtractorService()

    try:
        # Step 1: Full OCR
        print("\n3. Running Full OCR...")
        print("   This may take a few minutes depending on PDF size...")
        
        if use_local:
            ocr_result = await full_ocr_service.process_escritura(
                local_path=str(test_pdf_path)
            )
        else:
            ocr_result = await full_ocr_service.process_escritura(
                bucket=bucket,
                file_name=file_name
            )

        print(f"\n   ✓ OCR Complete:")
        print(f"     - Total pages: {ocr_result.total_pages}")
        print(f"     - Pages processed: {len(ocr_result.paginas)}")
        print(f"     - Chunks created: {len(ocr_result.chunks)}")

        # Step 2: Load inmuebles from CSV or use default
        print("\n4. Loading inmuebles...")
        
        if csv_path:
            print(f"   Reading CSV: {csv_path}")
            test_inmuebles = load_inmuebles_from_csv(csv_path)
            print(f"   ✓ Loaded {len(test_inmuebles)} inmueble(s) from CSV")
        else:
            print("   ⚠ No CSV provided, using default test inmueble")
            print("   💡 Tip: Use --csv path/to/inmuebles.csv to load from file")
            # Default test inmueble based on the PDF content
            test_inmuebles = [
                InmuebleInput(
                    inmueble="Casa Lote que hace parte de la urbanización El Hipódromo",
                    folio="370-404755"
                )
            ]

        print(f"\n   Searching for {len(test_inmuebles)} inmueble(s):")
        for inm in test_inmuebles:
            print(f"     - {inm.inmueble} (folio: {inm.folio})")

        # Step 3: Extract inmuebles
        print("\n5. Extracting inmuebles...")

        extraction_results, stats = await full_extractor_service.extract_inmuebles(
            ocr_result, test_inmuebles
        )

        # Step 4: Display results
        print("\n6. Results:")
        print("=" * 80)
        for i, result in enumerate(extraction_results, 1):
            print(f"\nInmueble {i}:")
            print(f"  Nombre: {result.inmueble}")
            print(f"  Folio: {result.folio}")
            print(f"  Status: {result.status}")
            if result.paginas:
                print(f"  Páginas: {result.paginas}")
            print(f"  Text length: {len(result.text)} chars")
            if result.status == "found":
                # Show first 200 chars of extracted text
                preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
                print(f"  Preview: {preview}")
            else:
                print(f"  Text: {result.text}")

        # Save results to JSON
        output_file = Path(__file__).parent.parent / "test_results.json"
        output_data = [result.model_dump() for result in extraction_results]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Display stats
        print(f"\n7. Statistics:")
        print(f"   Found by text-regex: {stats.found_by_text_regex}")
        print(f"   Found by refinement-llm: {stats.found_by_refinement_llm}")
        print(f"   Total found: {stats.total_found}/{len(test_inmuebles)}")
        print(f"   Total not found: {stats.total_not_found}")

        print(f"\n✓ Results saved to: {output_file}")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await full_ocr_service.close()
        await full_extractor_service.close()

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Full Process Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local file with CSV
  python scripts/test_full_process.py ../escritura_test.pdf --csv inmuebles.csv
  
  # Local file (absolute path) with CSV
  python scripts/test_full_process.py /path/to/escritura.pdf --csv /path/to/inmuebles.csv
  
  # GCS with CSV
  python scripts/test_full_process.py --bucket escrituras_publicas --file escritura_109668.pdf --csv inmuebles.csv
  
  # Default (tries local file, uses default test inmueble if no CSV)
  python scripts/test_full_process.py
  
  # CSV only (uses default PDF)
  python scripts/test_full_process.py --csv inmuebles.csv
        """
    )
    
    parser.add_argument(
        "local_path",
        nargs="?",
        default=None,
        help="Path to local PDF file (relative or absolute). If not provided and no --bucket, uses default."
    )
    
    parser.add_argument(
        "--bucket",
        type=str,
        help="GCS bucket name (requires --file)"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        dest="file_name",
        help="GCS file name (requires --bucket)"
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        dest="csv_path",
        help="Path to CSV file with inmuebles (columns: Inmueble, Folio)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.bucket and not args.file_name:
        parser.error("--bucket requires --file")
    if args.file_name and not args.bucket:
        parser.error("--file requires --bucket")
    if args.bucket and args.local_path:
        parser.error("Cannot use both local_path and GCS (--bucket/--file)")
    
    # Default to local file if nothing provided
    if not args.local_path and not args.bucket:
        default_path = Path(__file__).parent / "escritura_test.pdf"
        if default_path.exists():
            args.local_path = str(default_path)
            print(f"Using default local file: {args.local_path}")
        else:
            parser.error("No input provided and default file not found. Provide local_path or --bucket/--file")
    
    asyncio.run(test_full_process(
        local_path=args.local_path,
        bucket=args.bucket,
        file_name=args.file_name,
        csv_path=args.csv_path
    ))

