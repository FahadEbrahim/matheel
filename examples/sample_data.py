"""Generate the tiny sample Java files used by Matheel examples."""

import argparse
from pathlib import Path
from tempfile import gettempdir
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


DEFAULT_SAMPLE_ARCHIVE = Path(gettempdir()) / "matheel_examples" / "sample_pairs.zip"
CODE_A_NAME = "code_1.java"
CODE_B_NAME = "code_3_plag.java"
_ZIP_TIMESTAMP = (2024, 1, 1, 0, 0, 0)

SAMPLE_FILES = {
    "hello_world.java": """class Hello
{
    public static void main(String []args)
    {
        System.out.println("Hello World");
    }
};
""",
    CODE_A_NAME: """import java.util.Arrays;
import java.util.List;

public class max {
   public static void main(String[] args){
      List<Integer> arrayList = Arrays.asList(10, 5, 1, 20, 100);
      int max = Integer.MIN_VALUE;
      for (Integer i : arrayList) {
         if (i > max)
         max = i;
      }
      System.out.println(max);
   }
}
""",
    "code_2_plag.java": """import java.util.*;




public class ma {
public static void main(String[] args)
{List<Integer> a = Arrays.asList(10, 5, 1, 20, 100);
int m = Integer.MIN_VALUE;
for (Integer i : arrayList)
{if (i > m)
m = i;}
System.out.println(m);}}
""",
    CODE_B_NAME: """import java.util.Arrays;
import java.util.List;

public class maxi {
public static void main(String[] args)
{
    System.out.println();
    List<Integer> aa = Arrays.asList(10, 5, 1, 20, 100);
    int mm = Integer.MIN_VALUE;
    System.out.println();
    for (Integer i : aa)
    {if (i > mm)    mm = i;}
System.out.println(mm);
}}
""",
    "code_4_nonplag.java": """import java.util.*;
public class maxCollection{
   public static void main(String[] args){
      int array[] = {10, 5, 1, 20, 100};
      List<Integer> list = new ArrayList<>();
      for(int a=0;a<array.length;a++){
         list.add(array[a]);
      }
      System.out.println(Collections.max(list));
   }
}
""",
}


def sample_archive_path():
    return write_sample_archive(DEFAULT_SAMPLE_ARCHIVE, overwrite=True)


def write_sample_archive(output_path=DEFAULT_SAMPLE_ARCHIVE, overwrite=False):
    target = Path(output_path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"{target} already exists. Pass overwrite=True to replace it.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
        for name, text in SAMPLE_FILES.items():
            info = ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, text)
    return target


def write_sample_files(output_dir, overwrite=False):
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, text in SAMPLE_FILES.items():
        path = target / name
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists. Pass overwrite=True to replace it.")
        path.write_text(text, encoding="utf-8")
    return target


def load_sample_pair(archive_path=None):
    with ZipFile(archive_path or sample_archive_path()) as archive:
        code_a = archive.read(CODE_A_NAME).decode("utf-8")
        code_b = archive.read(CODE_B_NAME).decode("utf-8")
    return code_a, code_b


def main():
    args = _parse_args()
    output = Path(args.output)
    output_format = args.format or ("zip" if output.suffix == ".zip" else "directory")
    if output_format == "zip":
        path = write_sample_archive(output, overwrite=args.overwrite)
    else:
        path = write_sample_files(output, overwrite=args.overwrite)
    print(f"Wrote sample {output_format} to {path}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="sample_pairs.zip",
        help="Output archive or directory path.",
    )
    parser.add_argument(
        "--format",
        choices=("zip", "directory"),
        help="Output format. Defaults to zip for .zip paths and directory otherwise.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated sample files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
